# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify DSH profile lookup and prompt content."""

import pytest

from agentinfer.agentbench.agents.dsh.profiles import REQUIRED_ENDPOINT, get_profile
from agentinfer.agentbench.benchkit.dataset import Task


def _task() -> Task:
    return Task(
        "astropy__astropy-7166",
        "astropy/astropy",
        "abc123",
        "Fix the bug",
        fail_to_pass=("test_a",),
        pass_to_pass=("test_b",),
    )


def test_required_endpoint_is_openai_chat_completions() -> None:
    assert REQUIRED_ENDPOINT == "/v1/chat/completions"


def test_single_prompt_forbids_orchestration_and_questions() -> None:
    prompt = get_profile("single").build_prompt(_task())

    assert "Do not create subagents" in prompt
    assert "Do not ask the user questions" in prompt
    assert "astropy__astropy-7166" in prompt
    assert "FAIL_TO_PASS test cases" in prompt
    assert "test_a" in prompt
    assert "PASS_TO_PASS test cases" in prompt
    assert "test_b" in prompt


def test_single_profile_enforces_one_agent_tool_policy() -> None:
    profile = get_profile("single")

    assert profile.permission_mode == "workspace-write"
    for plugin in (
        "tool-subagent",
        "tool-subagent-fork",
        "tool-subagent-control",
        "tool-subagent-list-agents",
        "tool-subagent-report",
        "tool-workflow",
        "workflow-worker-thread",
        "tool-ralph",
        "tool-goal",
        "goal-round-driver",
        "command-goal",
        "goal",
        "tool-jobs",
        "jobs",
    ):
        assert f"- id: {plugin}\n  disabled: true" in profile.policy_patch
    assert "enableRunInBackground: false" in profile.policy_patch


def test_plan_subagent_profile_requires_read_only_exploration() -> None:
    profile = get_profile("plan-subagent")
    prompt = profile.build_prompt(_task())

    assert profile.enforce_plan_mode is True
    assert profile.permission_mode == "workspace-write"
    assert "first analyze the task" in prompt
    assert "planning phase" in prompt
    assert "`subagent` tool at least once" in prompt
    assert "Each subagent is read-only" in prompt
    assert "Do not delegate implementation" in prompt
    assert "Do not ask the user questions" in prompt
    assert "backgroundMode: one-shot" in profile.policy_patch
    assert "maxDepth: 1" in profile.policy_patch
    assert "You are a focused Explore subagent" in profile.policy_patch
    assert "    toolFilter:\n      allow:" in profile.policy_patch
    for tool in ("glob", "grep", "read"):
        assert f"        - {tool}" in profile.policy_patch
    assert "        - subagent\n" not in profile.policy_patch
    assert "        - bash\n" not in profile.policy_patch


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="profile"):
        get_profile("unknown")
