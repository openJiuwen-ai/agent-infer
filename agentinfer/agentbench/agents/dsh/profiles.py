# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Map configured DSH profile names to benchmark prompts.

'get_profile' is the runtime entry point. This module owns DSH prompt policy;
it does not launch DSH or manage its state.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ...benchkit.dataset import IMPLEMENTATION_CONSTRAINTS, Task, task_context

REQUIRED_ENDPOINT = "/v1/chat/completions"


@dataclass(frozen=True)
class DshProfile:
    """Bundle one runnable DSH headless profile and its enforced policy.

    ``enforce_plan_mode`` configures the task-local AgentBench bridge plugin
    that forces DSH plan mode on and auto-approves the exit_plan_mode review
    (headless has no UI transport, so the review has no human answerer).
    """

    name: str
    build_prompt: Callable[[Task], str]
    permission_mode: str
    policy_patch: str
    enforce_plan_mode: bool = False


_PERMISSION_MODE = "workspace-write"

_DISABLED_ORCHESTRATION_POLICY = """\
- id: tool-subagent-fork
  disabled: true
- id: tool-subagent-control
  disabled: true
- id: tool-subagent-list-agents
  disabled: true
- id: tool-subagent-report
  disabled: true
- id: tool-workflow
  disabled: true
- id: workflow-worker-thread
  disabled: true
- id: tool-ralph
  disabled: true
- id: tool-goal
  disabled: true
- id: goal-round-driver
  disabled: true
- id: command-goal
  disabled: true
- id: goal
  disabled: true
- id: tool-jobs
  disabled: true
- id: jobs
  disabled: true
- id: tool-bash
  config:
    enableRunInBackground: false
- id: tool-pwsh
  config:
    enableRunInBackground: false
"""

_SINGLE_POLICY_PATCH = (
    """\
- id: tool-subagent
  disabled: true
"""
    + _DISABLED_ORCHESTRATION_POLICY
)

_PLAN_SUBAGENT_POLICY_PATCH = (
    _DISABLED_ORCHESTRATION_POLICY
    + """\
- id: tool-subagent
  config:
    provider: spawn
    toolName: subagent
    enableRunInBackground: false
    backgroundMode: one-shot
    maxDepth: 1
    persona: >-
      You are a focused Explore subagent. Inspect and search the repository to answer the lead's question.
      Do not edit files, run mutating commands, create subagents, goals, workflows, or background jobs.
      Report concise evidence and recommendations to the lead.
    toolFilter:
      allow:
        - glob
        - grep
        - read
"""
)


def _single_prompt(task: Task) -> str:
    """Build the direct-execution prompt without DSH orchestration tools."""

    return f"""You are an AI coding agent. Solve this SWE-bench task.

{task_context(task)}
Work directly in the provided repository to implement the fix.

{IMPLEMENTATION_CONSTRAINTS}- Do not create subagents, goals, workflows, or background jobs; work directly.
- Do not ask the user questions or request approval; continue autonomously.
- When you are done, provide the final result and stop.
"""


def _plan_subagent_prompt(task: Task) -> str:
    """Build the prompt for lead planning with a focused DSH subagent."""

    return f"""You are the lead of a coding team solving this SWE-bench task.

{task_context(task)}
You must first analyze the task and formulate an implementation plan before investigating or implementing the fix.
During that planning phase:
- Use the `subagent` tool at least once to delegate focused repository exploration.
- Each subagent is read-only: ask it to inspect code, trace the root cause, or identify tests; do not ask it to edit files.
- Wait for every subagent report, synthesize the evidence, and finalize the implementation plan before editing.

After planning, write the final patch yourself. Do not delegate implementation.

{IMPLEMENTATION_CONSTRAINTS}- Do not ask the user questions or request approval; continue autonomously.
- When you are done, provide the final result and stop.
"""


SINGLE_PROFILE = DshProfile(
    "single",
    _single_prompt,
    _PERMISSION_MODE,
    _SINGLE_POLICY_PATCH,
)
PLAN_SUBAGENT_PROFILE = DshProfile(
    "plan-subagent",
    _plan_subagent_prompt,
    _PERMISSION_MODE,
    _PLAN_SUBAGENT_POLICY_PATCH,
    enforce_plan_mode=True,
)

_BUILTIN = {
    SINGLE_PROFILE.name: SINGLE_PROFILE,
    PLAN_SUBAGENT_PROFILE.name: PLAN_SUBAGENT_PROFILE,
}


def get_profile(profile_name: str) -> DshProfile:
    """Return the DSH profile selected by agent.profile."""

    try:
        return _BUILTIN[profile_name]
    except KeyError:
        raise ValueError(f"Unknown DSH agent profile: {profile_name!r}") from None
