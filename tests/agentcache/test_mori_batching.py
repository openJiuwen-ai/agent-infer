# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Native batch budgets preserve forwarding, limits, and failure settlement."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from agentinfer.agentcache.core.controller import build_mori_controller
from agentinfer.agentcache.mori.scheduler import (
    AgentCacheAsyncSchedulerBridge,
    AgentCacheSyncSchedulerBridge,
    _VllmAdmissionHooks,
)
from agentinfer.scheduling.mori import MORIConfig, MORIStrategy
from tests.agentcache.core.test_mori_controller import backend

pytestmark = pytest.mark.cpu_test


def request(*, output=1, computed=101, prompt=100):
    # Deliberately has no output limit, future length, or tool information.
    return SimpleNamespace(num_output_tokens=output, num_computed_tokens=computed, num_prompt_tokens=prompt)


def hooks(*, cap=2048):
    instance = object.__new__(_VllmAdmissionHooks)
    instance.controller = SimpleNamespace(strategy=MORIStrategy(MORIConfig(decode_batch_tokens=cap)))
    return instance


def test_batch_budget_configuration_is_explicit_and_defaults_off():
    assert MORIConfig().decode_batch_tokens == 0
    assert build_mori_controller(backend(), {}).strategy.config.decode_batch_tokens == 0
    controller = build_mori_controller(backend(), {"mori": {"decode_batch_tokens": 2048}})
    assert controller.strategy.config.decode_batch_tokens == 2048


@pytest.mark.parametrize("value", [-1, True, 1.5, "2048", None])
def test_batch_budget_configuration_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        MORIConfig(decode_batch_tokens=value)
    with pytest.raises(ValueError):
        build_mori_controller(backend(), {"mori": {"decode_batch_tokens": value}})


def test_disabled_feature_never_reads_native_request_or_budget():
    instance = hooks(cap=0)
    sentinel = object()
    native = Mock(return_value=sentinel)
    assert instance.schedule_native(object(), native, "argument", keyword=3) is sentinel
    native.assert_called_once_with("argument", keyword=3)
    assert not instance.controller.strategy.counters


@pytest.mark.parametrize(("ceiling", "chunked"), [(None, True), (True, True), (0, True), ("8192", True), (8192, False)])
def test_unsupported_native_budget_fails_before_installing_cache_hooks(ceiling, chunked):
    owner = SimpleNamespace(
        max_num_scheduled_tokens=ceiling,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked),
    )
    config = SimpleNamespace(
        additional_config={"agentcache": {"strategy": "mori", "mori": {"decode_batch_tokens": 2048}}}
    )
    with (
        patch.object(_VllmAdmissionHooks, "backend_pool_info", return_value=backend()),
        patch("agentinfer.agentcache.mori.scheduler.install_vllm_mori") as install,
        pytest.raises(ValueError, match="decode_batch_tokens requires"),
    ):
        _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=100, kv_cache_groups=(object(),)), 16)
    install.assert_not_called()


@pytest.mark.parametrize("running", [[], [request(output=0)], [request(computed=99)]])
def test_prefill_without_actual_output_keeps_full_native_budget(running):
    instance = hooks()
    owner = SimpleNamespace(running=running, max_num_scheduled_tokens=8192)

    def native():
        assert owner.max_num_scheduled_tokens == 8192
        return "unchanged"

    assert instance.schedule_native(owner, native) == "unchanged"
    assert not instance.controller.strategy.counters


@pytest.mark.parametrize(("cap", "ceiling", "count", "expected"), [(2048, 8192, 2, 2048), (1, 8192, 7, 7)])
def test_decoding_batch_is_bounded_and_keeps_a_slot_for_each_running_request(cap, ceiling, count, expected):
    instance = hooks(cap=cap)
    running = [request()] + [request(output=0) for _ in range(count - 1)]
    owner = SimpleNamespace(running=running, max_num_scheduled_tokens=ceiling)
    output = SimpleNamespace(num_scheduled_tokens={"decode": 1, "prefill": expected - 1})

    def native(*args, **kwargs):
        assert args == ("argument",) and kwargs == {"keyword": 3}
        assert owner.max_num_scheduled_tokens == expected
        assert owner.running is running
        return output

    assert instance.schedule_native(owner, native, "argument", keyword=3) is output
    assert owner.max_num_scheduled_tokens == ceiling
    assert instance.controller.strategy.counters["decode_batch_capped_iterations"] == 1
    assert instance.controller.strategy.counters["decode_batch_assigned_tokens"] == expected
    assert instance.controller.strategy.counters["decode_batch_max_assigned_tokens"] == expected


@pytest.mark.parametrize("cap", [2048, 8192])
def test_policy_cannot_raise_a_smaller_configured_native_ceiling(cap):
    instance = hooks(cap=cap)
    owner = SimpleNamespace(running=[request()], max_num_scheduled_tokens=1024)

    def native():
        assert owner.max_num_scheduled_tokens == 1024
        return "unchanged"

    assert instance.schedule_native(owner, native) == "unchanged"
    assert not instance.controller.strategy.counters


def test_native_failure_restores_budget_and_does_not_record_assigned_work():
    instance = hooks()
    owner = SimpleNamespace(running=[request()], max_num_scheduled_tokens=8192)

    def native():
        assert owner.max_num_scheduled_tokens == 2048
        raise RuntimeError("native schedule failed")

    with pytest.raises(RuntimeError, match="native schedule failed"):
        instance.schedule_native(owner, native)
    assert owner.max_num_scheduled_tokens == 8192
    assert instance.controller.strategy.counters == {"decode_batch_native_errors": 1}


def test_empty_native_output_earns_no_assigned_work():
    instance = hooks()
    owner = SimpleNamespace(running=[request()], max_num_scheduled_tokens=8192)
    output = SimpleNamespace(num_scheduled_tokens={})
    assert instance.schedule_native(owner, lambda: output) is output
    assert owner.max_num_scheduled_tokens == 8192
    assert not instance.controller.strategy.counters


def test_budget_returns_to_full_prefill_after_actual_decoder_leaves():
    instance = hooks()
    decoder, prefill = request(), request(output=0)
    owner = SimpleNamespace(running=[decoder, prefill], max_num_scheduled_tokens=8192)
    observed = []

    def native():
        observed.append(owner.max_num_scheduled_tokens)
        return SimpleNamespace(num_scheduled_tokens={"prefill": owner.max_num_scheduled_tokens - 1})

    instance.schedule_native(owner, native)
    owner.running.remove(decoder)
    instance.schedule_native(owner, native)
    assert observed == [2048, 8192]
    assert owner.running == [prefill]
    assert owner.max_num_scheduled_tokens == 8192


@pytest.mark.parametrize("fail", [False, True])
def test_decode_priority_preserves_native_removals_and_new_arrivals_on_return_or_failure(fail):
    instance = hooks()
    first, decoder, second, newcomer = request(output=0), request(), request(output=0), request(output=0)
    running = [first, decoder, second]
    owner = SimpleNamespace(running=running, max_num_scheduled_tokens=8192)

    def native():
        assert all(a is b for a, b in zip(owner.running, [decoder, first, second], strict=False))
        # Model genuine completion/preemption and admission by native scheduling.
        owner.running.pop(0)
        owner.running.append(newcomer)
        if fail:
            raise RuntimeError("native partial transition")
        return SimpleNamespace(num_scheduled_tokens={"prefill": 2048})

    if fail:
        with pytest.raises(RuntimeError, match="partial transition"):
            instance.schedule_native(owner, native)
    else:
        instance.schedule_native(owner, native)
        assert instance.controller.strategy.counters["decode_batch_reordered_iterations"] == 1
    assert owner.running is running
    assert len(running) == 3
    assert all(a is b for a, b in zip(running, [first, second, newcomer], strict=False))
    assert owner.max_num_scheduled_tokens == 8192


@pytest.mark.parametrize(
    ("bridge_type", "native_type"),
    [(AgentCacheAsyncSchedulerBridge, AsyncScheduler), (AgentCacheSyncSchedulerBridge, Scheduler)],
)
def test_both_native_bridges_apply_the_budget_and_restore_it_before_observation(bridge_type, native_type):
    instance = hooks()
    instance.before_schedule = Mock()
    bridge = object.__new__(bridge_type)
    bridge._agentcache = instance
    bridge.running = [request()]
    bridge.max_num_scheduled_tokens = 8192
    output = SimpleNamespace(num_scheduled_tokens={"decode": 1, "prefill": 2047})

    def native(owner):
        assert owner is bridge and owner.max_num_scheduled_tokens == 2048
        return output

    def observe(actual):
        assert actual is output and bridge.max_num_scheduled_tokens == 8192

    instance.observe_prefix_cache = Mock(side_effect=observe)
    with patch.object(native_type, "schedule", native):
        assert bridge.schedule() is output
    instance.before_schedule.assert_called_once()
    instance.observe_prefix_cache.assert_called_once_with(output)
