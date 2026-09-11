# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Verify benchmark-owned prompt fragments."""

from agentinfer.agentbench.agents.claude.profiles import get_profile as get_claude_profile
from agentinfer.agentbench.agents.dsh.profiles import get_profile as get_dsh_profile
from agentinfer.agentbench.agents.jiuwenswarm.profiles import get_profile as get_jiuwenswarm_profile
from agentinfer.agentbench.benchkit.dataset import IMPLEMENTATION_CONSTRAINTS, Task, task_context


def _task() -> Task:
    return Task(
        "astropy__astropy-7166",
        "astropy/astropy",
        "abc123",
        "Fix the bug",
        fail_to_pass=("test_a",),
        pass_to_pass=("test_b",),
    )


def test_shared_task_context_includes_benchmark_constraints() -> None:
    context = task_context(_task())

    assert "astropy__astropy-7166" in context
    assert "FAIL_TO_PASS test cases" in context
    assert "test_a" in context
    assert "PASS_TO_PASS test cases" in context
    assert "test_b" in context
    assert "Install missing dependencies only inside the task workspace" in IMPLEMENTATION_CONSTRAINTS


def test_runnable_profiles_reuse_shared_task_context_and_constraints() -> None:
    task = _task()
    for profile in (
        get_claude_profile("single"),
        get_claude_profile("plan-subagent"),
        get_dsh_profile("single"),
        get_dsh_profile("plan-subagent"),
        get_jiuwenswarm_profile("code.normal"),
    ):
        prompt = profile.build_prompt(task)
        assert task_context(task) in prompt
        assert IMPLEMENTATION_CONSTRAINTS in prompt
