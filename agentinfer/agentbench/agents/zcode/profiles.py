# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Prompts and tool policy for the two ZCode benchmark profiles."""

from dataclasses import dataclass

from ...benchkit.dataset import IMPLEMENTATION_CONSTRAINTS, Task, task_context

EXPLORE_TOOL_ROUNDS = 8


@dataclass(frozen=True)
class ZCodeProfile:
    name: str
    planning: bool = False
    supports_subagent_limits: bool = False

    def build_prompt(
        self, task: Task, *, explore: str, max_subagents: int = 1, max_concurrent_subagents: int = 1
    ) -> str:
        policy = (
            f"Your first tool call must be Agent with subagent_type={explore} and run_in_background=false.\n"
            f"Delegate at least one focused, read-only exploration, at most {max_subagents} total "
            f"and {max_concurrent_subagents} concurrently, using the agent type specified above.\n"
            "Within those limits, launch independent explorations together as multiple Agent calls in the same response. "
            "Then wait for the whole batch; use successive batches only when needed.\n"
            "Ask the child to use Glob, Grep and Read, read at most 200 lines per call, and return a short report.\n"
            "Wait for all accepted explorations to finish. If more evidence is needed and budget remains, delegate again; "
            "otherwise call ExitPlanMode with a concise implementation plan. You need not use all available slots.\n"
            "After approval, implement and test the fix yourself; do not delegate implementation.\n"
            if self.planning
            else "Work directly; do not create subagents or enter plan mode.\n"
        )
        return (
            f"Solve this SWE-bench coding task.\n\n{task_context(task)}\n"
            f"{policy}\n{IMPLEMENTATION_CONSTRAINTS}"
            "- Listed FAIL_TO_PASS/PASS_TO_PASS cases may be added only by the evaluator and absent from this checkout.\n"
            "- Validate the issue's reproduction and available tests, then finish; do not keep searching for or inventing evaluator-only cases.\n"
            "- Do not ask the user questions or start background work.\n"
            "- If tests fail because dependencies are missing, report that blocker; do not repeatedly generate verification scripts.\n"
            "- Finish with a concise report after implementing and testing the fix.\n"
        )

    @property
    def disallowed_tools(self) -> str:
        tools = "AskUserQuestion EnterPlanMode WebSearch WebFetch"
        return tools if self.planning else f"{tools} Agent Task ExitPlanMode"


PROFILES = {
    "single": ZCodeProfile("single"),
    "plan-subagent": ZCodeProfile("plan-subagent", planning=True, supports_subagent_limits=True),
}


def explore_name(session_id: str) -> str:
    """Keep task-owned tools independent of repository-provided agent profiles."""
    return f"agentbench-explore-{session_id}"


def get_profile(name: str) -> ZCodeProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown ZCode agent profile: {name!r}") from None
