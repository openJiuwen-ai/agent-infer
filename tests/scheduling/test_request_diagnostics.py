# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Slow-request phase attribution, bounded output, and lifecycle cleanup."""

import json
import logging
from unittest.mock import Mock, patch

import pytest

from agentinfer.scheduling.domain import ProgramRef
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MORIConfig, build_mori_strategy
from agentinfer.scheduling.observability import SchedulerObservabilityConfig
from agentinfer.scheduling.progress_ttl import build_progress_ttl_strategy
from agentinfer.scheduling.request_diagnostics import RequestDiagnostics
from agentinfer.scheduling.runtime import ProgramScheduler
from tests.scheduling.test_mori_waiting import pool

pytestmark = pytest.mark.cpu_test


def rows(log):
    return [json.loads(call.args[1]) for call in log.warning.call_args_list]


def tracker(**kwargs):
    log = Mock(spec=logging.Logger)
    return RequestDiagnostics(logger=log, **kwargs), log


def test_normal_requests_do_no_log_or_context_work_and_leave_no_history():
    context = Mock(side_effect=AssertionError("normal request evaluated context"))
    diagnostics, log = tracker(context_provider=context)
    for index in range(1000):
        request_id = str(index)
        diagnostics.arrive(request_id, ProgramRef("p", index), 0, prompt_tokens=32)
        diagnostics.admitted(request_id, 1)
        diagnostics.native_execution(request_id, 2, running=True)
        diagnostics.first_output(request_id, 3)
        diagnostics.finish(request_id, 4)
    diagnostics.poll(100)
    assert diagnostics.active_count == 0
    assert not diagnostics._program_requests
    log.warning.assert_not_called()
    context.assert_not_called()


def test_slow_phase_timing_retains_initial_schedule_and_reports_preemption():
    diagnostics, log = tracker()
    diagnostics.arrive("r", ProgramRef("branch", 2), 0, external_program_id="external", prompt_tokens=96)
    diagnostics.poll(30)
    diagnostics.admitted("r", 31)
    diagnostics.poll(90)
    diagnostics.native_execution("r", 95, running=True)
    for now in range(96, 100):
        diagnostics.native_execution("r", now, running=True)
    diagnostics.poll(150)
    diagnostics.native_execution("r", 151, running=False)
    diagnostics.native_execution("r", 152, running=False)
    diagnostics.poll(210)
    diagnostics.native_execution("r", 220, running=True)
    diagnostics.cache_observation("r", cached_prefix_tokens=80, hbm_cached_prefix_tokens=32)
    diagnostics.first_output("r", 230)
    diagnostics.first_output("r", 231)
    diagnostics.finish("r", 240)
    events = rows(log)
    assert [row["phase"] for row in events] == [
        "request_pool",
        "native_waiting",
        "native_prefill",
        "native_waiting",
        "native_decode",
        "native_decode",
    ]
    final = events[-1]
    assert final["request_pool_seconds"] == 31
    assert final["native_wait_seconds"] == 64
    assert final["scheduled_to_first_output_seconds"] == 135  # includes the 69-second preemption
    assert final["decode_seconds"] == 10
    assert final["native_preemptions"] == 1
    assert final["cached_prefix_tokens"] == 80
    assert final["hbm_cached_prefix_tokens"] == 32
    assert final["external_program_id"] == "external"
    assert final["generation"] == 2
    assert final["trigger"] == "completed"
    assert diagnostics.active_count == 0


def test_missing_native_observation_is_unknown_instead_of_fabricated_prefill():
    diagnostics, log = tracker()
    diagnostics.arrive("r", ProgramRef("p", 0), 0)
    diagnostics.admitted("r", 10)
    diagnostics.first_output("r", 40)
    final = rows(log)[0]
    assert final["first_native_at_monotonic_s"] is None
    assert final["native_wait_seconds"] is None
    assert final["scheduled_to_first_output_seconds"] is None


def test_fast_first_output_remains_quiet_during_and_after_a_long_decode():
    context = Mock(side_effect=AssertionError("fast TTFT evaluated context"))
    diagnostics, log = tracker(context_provider=context)
    diagnostics.arrive("r", ProgramRef("p", 0), 0)
    diagnostics.admitted("r", 1)
    diagnostics.native_execution("r", 2, running=True)
    diagnostics.first_output("r", 3)
    for now in range(30, 300, 5):
        diagnostics.poll(now)
    diagnostics.finish("r", 300)
    assert diagnostics.active_count == 0
    log.warning.assert_not_called()
    context.assert_not_called()


def test_slow_first_output_gets_one_recovery_and_one_finish_summary_without_decode_polls():
    diagnostics, log = tracker()
    diagnostics.arrive("r", ProgramRef("p", 0), 0)
    diagnostics.admitted("r", 1)
    diagnostics.native_execution("r", 2, running=True)
    diagnostics.first_output("r", 40)
    for now in range(45, 300, 5):
        diagnostics.poll(now)
    diagnostics.finish("r", 300)
    assert [row["trigger"] for row in rows(log)] == ["first_output", "completed"]
    assert rows(log)[-1]["ttft_seconds"] == 40
    assert rows(log)[-1]["decode_seconds"] == 260


def test_poll_caps_rows_throttles_repeats_and_eventually_visits_all_live_requests():
    diagnostics, log = tracker(max_events_per_poll=3)
    for index in range(19):
        diagnostics.arrive(str(index), ProgramRef("p", index), 0)
    for now in range(30, 65, 5):
        before = log.warning.call_count
        diagnostics.poll(now)
        assert log.warning.call_count - before <= 3
        diagnostics.poll(now + 1)
        assert log.warning.call_count - before <= 3
    assert {row["request_id"] for row in rows(log)} == {str(index) for index in range(19)}
    assert len(rows(log)) == 19
    diagnostics.poll(90)
    assert log.warning.call_count == 22


def test_scan_bound_rotates_past_many_young_requests_without_starving_an_old_tail():
    diagnostics, log = tracker()
    for index in range(512):
        diagnostics.arrive(str(index), ProgramRef("young", index), 99)
    diagnostics.arrive("old", ProgramRef("old", 0), 0)
    diagnostics.poll(100)
    diagnostics.poll(105)
    log.warning.assert_not_called()
    diagnostics.poll(110)
    assert [row["request_id"] for row in rows(log)] == ["old"]


def test_cancel_release_and_reused_request_ids_do_not_cross_generations():
    diagnostics, log = tracker()
    old, new = ProgramRef("p", 0), ProgramRef("p", 1)
    diagnostics.arrive("cancel", old, 0)
    diagnostics.finish("cancel", 40, outcome="cancelled")
    diagnostics.finish("cancel", 50, outcome="cancelled")
    diagnostics.arrive("old", old, 0)
    diagnostics.arrive("new", new, 0)
    diagnostics.release(old, 50)
    assert diagnostics.active_count == 1
    assert {row["trigger"] for row in rows(log)} == {"cancelled", "released"}
    diagnostics.release(new, 51)
    diagnostics.arrive("old", new, 60)
    diagnostics.poll(61)
    assert diagnostics.active_count == 1
    assert len(rows(log)) == 3


def test_warning_json_is_bounded_single_line_and_context_failures_do_not_escape():
    context = Mock(return_value={str(i): ["\n\U0001f600" * 300] * 100 for i in range(100)})
    diagnostics, log = tracker(context_provider=context)
    diagnostics.arrive("r\n" * 300, ProgramRef("p\n" * 300, 0), 0, external_program_id="x\n" * 300)
    diagnostics.poll(30)
    encoded = log.warning.call_args.args[1]
    assert len(encoded) <= 8192
    assert "\n" not in encoded
    assert json.loads(encoded)["context"] == {"truncated": True}
    context.side_effect = RuntimeError("private prompt must not be logged")
    diagnostics.poll(90)
    assert rows(log)[-1]["context"] == {"context_error": "RuntimeError"}
    assert "private prompt" not in log.warning.call_args.args[1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("threshold_seconds", 0),
        ("threshold_seconds", float("nan")),
        ("repeat_seconds", -1),
        ("poll_interval_seconds", float("inf")),
        ("max_events_per_poll", 0),
        ("max_events_per_poll", True),
    ],
)
def test_invalid_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        RequestDiagnostics(**{field: value})


def make_runtime(monkeypatch, diagnostics):
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    monkeypatch.setattr(components.strategy, "request_diagnostics", lambda: diagnostics)
    return ProgramScheduler(
        components.strategy,
        components.initial_factors,
        pool(),
        observability=SchedulerObservabilityConfig(enabled=False),
    )


def test_runtime_captures_real_queue_handoff_native_and_output_times_with_debug_disabled(monkeypatch):
    diagnostics, log = tracker()
    scheduler = make_runtime(monkeypatch, diagnostics)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert not scheduler.on_request_arrival("r", AgentIdentity("p"), 96, pool(hbm=0), "request")
        clock.return_value = 30
        scheduler.schedule_cycle(pool(hbm=0))
        assert rows(log)[-1]["phase"] == "request_pool"
        clock.return_value = 31
        scheduler.schedule_cycle(pool())
        assert len(scheduler.consume_admitted_requests()) == 1
        clock.return_value = 40
        scheduler.on_native_execution("r", running=True)
        scheduler.on_prefix_cache_observation("r", 80, hbm_cached_prefix_tokens=32)
        clock.return_value = 50
        scheduler.on_stream_output("r", 1)
        scheduler.on_stream_output("r", 2)
        clock.return_value = 60
        scheduler.on_request_completion("r", 98)
    final = rows(log)[-1]
    assert final["request_pool_seconds"] == 31
    assert final["native_wait_seconds"] == 9
    assert final["scheduled_to_first_output_seconds"] == 10
    assert final["decode_seconds"] == 10
    assert final["prompt_tokens"] == 96
    assert final["cached_prefix_tokens"] == 80
    assert diagnostics.active_count == 0


def test_runtime_polls_after_planner_and_cancels_only_matching_overlap(monkeypatch):
    diagnostics, log = tracker()
    scheduler = make_runtime(monkeypatch, diagnostics)
    seen = []
    original_resume = scheduler.strategy.schedule_resume

    def resume(*args):
        seen.append("planner")
        original_resume(*args)

    monkeypatch.setattr(scheduler.strategy, "schedule_resume", resume)
    diagnostics._context_provider = lambda *_: {"planner_ran": bool(seen)}
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert scheduler.on_request_arrival("first", AgentIdentity("external"), 96, pool(), "first")
        assert not scheduler.on_request_arrival("second", AgentIdentity("external"), 96, pool(), "second")
        second = scheduler.program_ref_for_request("second")
        assert second != scheduler.program_ref_for_request("first")
        clock.return_value = 30
        scheduler.schedule_cycle(pool())
        assert rows(log)[0]["context"]["planner_ran"]
        scheduler.cancel_request("second")
        assert diagnostics.active_count == 1
        assert rows(log)[-1]["program_id"] == second.program_id
        assert rows(log)[-1]["external_program_id"] == "external"
        scheduler.cancel_request("first")
    assert diagnostics.active_count == 0
    assert not diagnostics._program_requests


def test_completion_without_stream_observation_does_not_invent_a_first_token_timestamp(monkeypatch):
    diagnostics, log = tracker()
    scheduler = make_runtime(monkeypatch, diagnostics)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert scheduler.on_request_arrival("r", AgentIdentity("p"), 96, pool(), "r")
        clock.return_value = 2
        scheduler.on_native_execution("r", running=True)
        clock.return_value = 40
        scheduler.on_request_completion("r", 100)
    final = rows(log)[-1]
    assert final["first_output_observation"] == "missing_at_completion"
    assert final["first_output_at_monotonic_s"] is None
    assert final["ttft_seconds"] is None
    assert final["decode_seconds"] is None
    assert diagnostics.active_count == 0


def test_other_strategies_have_no_tracker():
    components = build_progress_ttl_strategy()
    scheduler = ProgramScheduler(components.strategy, components.initial_factors, pool())
    assert scheduler._request_diagnostics is None
