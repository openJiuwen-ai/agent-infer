"""Map configured Claude profile names to prompts and execution policy.

``get_profile`` is the runtime entry point. This module owns Claude-specific
tool, permission, topology, and prompt policy; it does not launch Claude or
manage terminal interaction.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ...benchkit.dataset import Task


@dataclass(frozen=True)
class ToolPolicy:
    """List Claude tools unavailable to a configured profile."""

    disallowed_tools: tuple[str, ...]


@dataclass(frozen=True)
class TopologyExpectation:
    """Describe the agent count and roles expected from a Claude profile."""

    min_agents: int
    max_agents: int | None
    expected_roles: frozenset[str]


@dataclass(frozen=True)
class ClaudeProfile:
    """Bundle the prompt and Claude runtime policy selected by profile name."""

    name: str
    build_prompt: Callable[[Task], str]
    tool_policy: ToolPolicy
    permission_mode: str
    expected_topology: TopologyExpectation
    interaction_profile: str


def _task_context(task: Task) -> str:
    """Render task identity, problem, and test constraints shared by prompts."""

    prompt = f"""Instance: {task.instance_id}
Repository: {task.repo}
Base commit: {task.base_commit}

Problem statement:
{task.problem_statement}
"""
    if task.fail_to_pass:
        prompt += "FAIL_TO_PASS test cases (the fix should make these pass):\n" + "\n".join(task.fail_to_pass) + "\n"
    if task.pass_to_pass:
        prompt += "PASS_TO_PASS test cases (must not break these):\n" + "\n".join(task.pass_to_pass) + "\n"
    return prompt


def _plan_subagent_prompt(task: Task) -> str:
    """Build the prompt for lead-driven planning with focused subagents."""

    return f"""You are the lead of a coding team solving this SWE-bench task.

{_task_context(task)}
You must call EnterPlanMode before investigating or implementing the fix.
While in plan mode:
- Use at least one focused Explore subagent to trace the root cause and inspect relevant code.
- Avoid spawning extra subagents unless they are clearly needed.
- Subagents may inspect, search, review, and run tests, but cannot edit files.
- Wait for all subagent reports, finalize the implementation plan, and call ExitPlanMode promptly.
- Plan approval is automatic in this benchmark; continue immediately with implementation.

After exiting plan mode, write the final patch yourself. Do not create an Agent Team.
Test the fix when practical and summarize the changes.

Constraints:
- Inspect the code first to understand the issue.
- Make minimal changes to fix the problem.
- Run the existing tests to ensure nothing is broken.
- Implement the fix by editing files in the repository.
- Do not only print or describe a patch; modify the files directly.
- When you are done, stop and wait at the prompt.
"""


def _single_prompt(task: Task) -> str:
    """Build the prompt for direct execution without subagents."""

    return f"""You are an AI coding agent. Solve this SWE-bench task.

{_task_context(task)}
Do not create subagents. Work directly in the repository to implement the fix.

Constraints:
- Inspect the code first to understand the issue.
- Make minimal changes to fix the problem.
- Run the existing tests to ensure nothing is broken.
- Implement the fix by editing files in the repository.
- Do not only print or describe a patch; modify the files directly.
- When you are done, stop and wait at the prompt.
"""


SINGLE_PROFILE = ClaudeProfile(
    name="single",
    build_prompt=_single_prompt,
    tool_policy=ToolPolicy(disallowed_tools=("Agent", "WebSearch", "WebFetch")),
    permission_mode="acceptEdits",
    expected_topology=TopologyExpectation(
        min_agents=1,
        max_agents=1,
        expected_roles=frozenset({"lead"}),
    ),
    interaction_profile="single",
)

PLAN_SUBAGENT_PROFILE = ClaudeProfile(
    name="plan-subagent",
    build_prompt=_plan_subagent_prompt,
    tool_policy=ToolPolicy(
        disallowed_tools=(
            "TeamCreate",
            "TeamDelete",
            "SendMessage",
            "WebSearch",
            "WebFetch",
        )
    ),
    permission_mode="bypassPermissions",
    expected_topology=TopologyExpectation(
        min_agents=2,
        max_agents=None,
        expected_roles=frozenset({"lead", "subagent"}),
    ),
    interaction_profile="plan-subagent",
)

_BUILTIN = {
    SINGLE_PROFILE.name: SINGLE_PROFILE,
    PLAN_SUBAGENT_PROFILE.name: PLAN_SUBAGENT_PROFILE,
}


def get_profile(profile_name: str) -> ClaudeProfile:
    """Return the Claude profile selected by ``agent.profile``."""

    try:
        return _BUILTIN[profile_name]
    except KeyError:
        raise ValueError(f"Unknown agent profile: {profile_name!r}") from None
