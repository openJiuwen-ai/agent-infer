# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Retirement must forget scheduling history without ending business relations."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from agentinfer.scheduling.domain import ProgramState, ProgramStatus
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.lifecycle import ProgramLifecycle
from agentinfer.scheduling.mori import MORITier
from agentinfer.scheduling.program_registry import ProgramRegistry
from agentinfer.scheduling.transitions import TransitionController, TransitionKind, TransitionRequest
from tests.scheduling.test_mori_branches import runtime

pytestmark = pytest.mark.cpu_test


def test_retirement_releases_owner_and_old_receipt_then_restarts_empty_history():
    scheduler, pool = runtime()
    identity = AgentIdentity("agent", lifecycle_token="old-receipt")
    assert scheduler.on_request_arrival("old", identity, 100, pool, "old")
    old = scheduler.program_ref_for_request("old")
    scheduler.on_request_completion("old", 120)
    assert scheduler._apply_transition(TransitionRequest(TransitionKind.PAUSE, old, "test_pause")).applied
    factors = scheduler.strategy_factors.for_program(old)
    scheduler.strategy_factors.set_program_factors(
        old, replace(factors, tier=MORITier.WAITING, cycles=((5.0, 1000.0),))
    )
    before = scheduler._snapshot()
    transitions = TransitionController(scheduler._apply_transition)

    result = transitions.retire(old, reason="test_retire")

    assert result.applied
    assert scheduler.registry.current_ref("agent") is None
    assert scheduler.strategy_factors.for_program(old) is None
    assert old not in scheduler._last_request_finished_at
    assert old not in scheduler._shared_prefix_freshness_anchor_at
    assert old not in scheduler._latest_lifecycle_tokens
    assert "old-receipt" not in scheduler._lifecycle_owners
    assert not scheduler._request_branches.family("agent")
    assert scheduler._advance_snapshot(before, transitions.take_applied_requests()).programs == ()
    assert scheduler.on_request_arrival("new", AgentIdentity("agent", lifecycle_token="new-receipt"), 200, pool, "new")
    new = scheduler.program_ref_for_request("new")
    assert new.generation == old.generation + 1
    assert scheduler.registry.require(new).step_count == 0
    assert scheduler.registry.require(new).tokens.estimated_context_tokens == 200
    assert scheduler.strategy_factors.for_program(new).cycles == ()
    assert not scheduler.on_scoped_response_completion("agent", ProgramLifecycle.TERMINAL, "old-receipt")
    assert not transitions.retire(old, reason="late_retire").applied
    assert scheduler.registry.get(new) is not None
    scheduler.on_request_completion("new", 220)
    assert transitions.pause(new, reason="test_pause").applied
    assert transitions.retire(new, reason="test_retire").applied
    assert not scheduler.on_scoped_response_completion("agent", ProgramLifecycle.TERMINAL, "old-receipt")
    assert "agent" in scheduler.registry._retired_metadata
    assert scheduler.on_scoped_response_completion("agent", ProgramLifecycle.TERMINAL, "new-receipt")


def test_retirement_cannot_erase_accepted_work_even_with_incorrect_status_observation():
    scheduler, pool = runtime()
    assert scheduler.on_request_arrival("request", AgentIdentity("agent"), 100, pool, "request")
    ref = scheduler.program_ref_for_request("request")
    program = scheduler.registry.require(ref)
    program.status = ProgramStatus.ACTING
    program.state = ProgramState.PAUSED

    result = scheduler._apply_transition(TransitionRequest(TransitionKind.RETIRE, ref, "unsafe"))

    assert not result.applied and result.reason == "retire_requires_idle"
    assert scheduler.program_ref_for_request("request") == ref
    assert scheduler.registry.get(ref) is program


@pytest.mark.parametrize(
    ("state", "status"),
    [(ProgramState.ACTIVE, ProgramStatus.ACTING), (ProgramState.PAUSED, ProgramStatus.REASONING)],
)
def test_retirement_requires_paused_acting(state, status):
    scheduler, _ = runtime()
    program, _ = scheduler.registry.materialize(AgentIdentity("agent"), initial_state=state)
    program.status = status

    result = scheduler._apply_transition(TransitionRequest(TransitionKind.RETIRE, program.ref, "unsafe"))

    assert not result.applied and result.reason == "retire_requires_paused_acting"
    assert scheduler.registry.get(program.ref) is program


@pytest.mark.parametrize("kind", [TransitionKind.ADMIT, TransitionKind.RESUME])
def test_admission_event_captures_wait_before_runtime_clears_start(kind):
    scheduler, pool = runtime(512)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=10.0) as clock:
        assert scheduler.on_request_arrival("resident", AgentIdentity("resident"), 400, pool, "resident")
        assert not scheduler.on_request_arrival("waiting", AgentIdentity("waiting"), 100, pool, "waiting")
        ref = scheduler.program_ref_for_request("waiting")
        snapshot = scheduler._snapshot()
        clock.return_value = 19.0
        request = TransitionRequest(kind, ref, "test_admit", backend_id=snapshot.backend_id)
        result = scheduler._apply_transition(request)

    assert result.applied
    assert result.event.field("scheduler_wait_seconds") == 9.0
    assert scheduler.registry.require(ref).wait_started_at_monotonic_s is None
    assert scheduler._advance_snapshot(snapshot, (request,)).program(ref).wait_started_at_monotonic_s is None


def test_retirement_preserves_parent_child_business_state_across_fresh_generations():
    registry = ProgramRegistry()
    parent, _ = registry.materialize(AgentIdentity("parent", task_id="task"))
    child, _ = registry.materialize(
        AgentIdentity("child", task_id="task", parent_program_id="parent", blocks_parent=True, agent_role="worker")
    )
    child.step_count = 7
    registry.retire(child.ref)
    assert parent.blocked_on_child
    assert registry.children(parent.ref) == (child.ref,)
    registry.retire(parent.ref)
    assert registry.views() == ()

    new_parent, _ = registry.materialize(AgentIdentity("parent"))
    new_child, _ = registry.materialize(AgentIdentity("child"))

    assert new_parent.ref.generation == parent.ref.generation + 1
    assert new_child.ref.generation == child.ref.generation + 1
    assert new_child.step_count == 0
    assert new_child.parent_program_id == "parent" and new_child.blocks_parent
    assert new_child.agent_role == "worker"
    assert new_parent.blocked_on_child
    assert registry.children(new_parent.ref) == (new_child.ref,)
    assert set(registry.task_programs("task")) == {new_parent.ref, new_child.ref}
    assert not registry._retired_metadata


def test_terminal_parent_release_detaches_retired_child_business_relation():
    registry = ProgramRegistry()
    parent, _ = registry.materialize(AgentIdentity("parent"))
    child, _ = registry.materialize(AgentIdentity("child", parent_program_id="parent", blocks_parent=True))
    registry.retire(child.ref)
    registry.release(parent.ref)

    new_child, _ = registry.materialize(AgentIdentity("child"))

    assert new_child.parent_program_id is None
    assert not new_child.blocks_parent


@pytest.mark.parametrize("scoped", [False, True])
def test_terminal_response_can_finish_retired_business_metadata(scoped):
    scheduler, pool = runtime()
    parent, _ = scheduler.registry.materialize(AgentIdentity("parent"), initial_state=ProgramState.ACTIVE)
    parent.status = ProgramStatus.ACTING
    identity = AgentIdentity("child", parent_program_id="parent", blocks_parent=True, lifecycle_token="old")
    assert scheduler.on_request_arrival("request", identity, 100, pool, "request")
    child = scheduler.program_ref_for_request("request")
    scheduler.on_request_completion("request", 120)
    assert scheduler._apply_transition(TransitionRequest(TransitionKind.PAUSE, child, "test_pause")).applied
    assert scheduler._apply_transition(TransitionRequest(TransitionKind.RETIRE, child, "test_retire")).applied
    assert parent.blocked_on_child

    assert not scheduler.on_scoped_response_completion("child", ProgramLifecycle.TERMINAL, None)
    assert not scheduler.on_scoped_response_completion("child", ProgramLifecycle.TERMINAL, "wrong")
    if scoped:
        assert scheduler.on_scoped_response_completion("child", ProgramLifecycle.TERMINAL, "old")
    else:
        assert scheduler.on_response_completion("child", ProgramLifecycle.TERMINAL)

    assert not parent.blocked_on_child
    assert "child" not in scheduler.registry._retired_metadata
    assert not scheduler.on_response_completion("child", ProgramLifecycle.TERMINAL)


def test_retired_terminal_cleanup_cannot_remove_a_reentered_generation():
    registry = ProgramRegistry()
    old, _ = registry.materialize(AgentIdentity("agent", task_id="task"))
    registry.retire(old.ref)
    new, _ = registry.materialize(AgentIdentity("agent"))

    assert not registry.release_retired("agent")
    assert registry.get(new.ref) is new
    assert registry.task_programs("task") == (new.ref,)


@pytest.mark.parametrize("cancelled", [False, True])
def test_lazy_demotion_retires_immediately_when_inference_finishes_or_is_cancelled(cancelled):
    scheduler, pool = runtime(1024)
    backend = pool.backends[0]
    pool = replace(pool, backends=(replace(backend, dp_ranks=(replace(backend.dp_ranks[0], total_dram_kv_tokens=0),)),))
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert scheduler.on_request_arrival("first", AgentIdentity("a-first"), 400, pool, "first")
        assert scheduler.on_request_arrival("second", AgentIdentity("z-second"), 400, pool, "second")
        ref = scheduler.program_ref_for_request("first")
        clock.return_value = 1.0
        scheduler.on_stream_output("first", 50)
        scheduler.schedule_cycle(pool)
        assert scheduler.registry.require(ref).marked_for_pause
        assert scheduler.strategy_factors.for_program(ref).pending_tier is MORITier.WAITING
        assert scheduler.strategy_factors.for_program(ref).tier is MORITier.GPU

        if cancelled:
            scheduler.cancel_request("first")
        else:
            scheduler.on_request_completion("first", 450)

        # No intervening periodic cycle or new arrival may observe the old owner.
        assert scheduler.registry.get(ref) is None
        assert scheduler.strategy_factors.for_program(ref) is None
        assert scheduler.registry.require(scheduler.program_ref_for_request("second")).status is ProgramStatus.REASONING
