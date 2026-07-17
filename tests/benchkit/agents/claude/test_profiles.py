"""Verify Claude profile lookup, policy, topology, and prompts."""

import pytest

from agentcache.benchmarks.agents.claude import (
    PLAN_SUBAGENT_PROFILE,
    SINGLE_PROFILE,
    get_profile,
)
from agentcache.benchmarks.benchkit.dataset import Task


def _task() -> Task:
    return Task(
        instance_id="task-a",
        repo="owner/repo",
        base_commit="abc123",
        problem_statement="Fix the bug",
        fail_to_pass=("tests/test_bug.py::test_fix",),
        pass_to_pass=("tests/test_existing.py::test_ok",),
    )


def test_get_profile_returns_supported_profiles() -> None:
    assert get_profile("single") is SINGLE_PROFILE
    assert get_profile("plan-subagent") is PLAN_SUBAGENT_PROFILE


def test_get_profile_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown agent profile: 'team'"):
        get_profile("team")


def test_single_profile_requires_one_lead_and_no_subagents() -> None:
    profile = SINGLE_PROFILE

    assert profile.permission_mode == "acceptEdits"
    assert profile.expected_topology.min_agents == 1
    assert profile.expected_topology.max_agents == 1
    assert profile.expected_topology.expected_roles == {"lead"}
    assert "Agent" in profile.tool_policy.disallowed_tools

    prompt = profile.build_prompt(_task())
    assert "Do not create subagents" in prompt
    assert "tests/test_bug.py::test_fix" in prompt
    assert "tests/test_existing.py::test_ok" in prompt


def test_plan_subagent_profile_requires_explore_without_agent_team() -> None:
    profile = PLAN_SUBAGENT_PROFILE

    assert profile.permission_mode == "bypassPermissions"
    assert profile.expected_topology.min_agents == 2
    assert profile.expected_topology.max_agents is None
    assert profile.expected_topology.expected_roles == {"lead", "subagent"}
    assert {"TeamCreate", "TeamDelete", "SendMessage"} <= set(profile.tool_policy.disallowed_tools)

    prompt = profile.build_prompt(_task())
    assert "call EnterPlanMode" in prompt
    assert "focused Explore subagent" in prompt
    assert "Do not create an Agent Team" in prompt
