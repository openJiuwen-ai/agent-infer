# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project

import pytest

from agentcache.core.scheduler import AgentScheduler
from vllm.config.scheduler import SchedulerConfig
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.scheduler import Scheduler as VllmScheduler

pytestmark = pytest.mark.cpu_test


def test_agent_scheduler_is_subclass_of_vllm_scheduler():
    assert issubclass(AgentScheduler, VllmScheduler)


def test_agent_scheduler_is_subclass_of_scheduler_interface():
    assert issubclass(AgentScheduler, SchedulerInterface)


def test_get_scheduler_cls_resolves_agent_scheduler():
    cfg = SchedulerConfig.default_factory(
        max_model_len=8192,
        scheduler_cls="agentcache.core.scheduler.AgentScheduler",
    )
    cls = cfg.get_scheduler_cls()
    assert cls is AgentScheduler


def test_get_scheduler_cls_default_is_vllm_scheduler():
    cfg = SchedulerConfig.default_factory(max_model_len=8192)
    cls = cfg.get_scheduler_cls()
    assert cls is VllmScheduler


def test_patch_sets_scheduler_cls_when_none(engine_args_patched):
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(model="test-model")
    assert args.scheduler_cls == "agentcache.core.scheduler.AgentScheduler"


def test_patch_preserves_explicit_scheduler_cls(engine_args_patched):
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(
        model="test-model",
        scheduler_cls="vllm.v1.core.sched.scheduler.Scheduler",
    )
    assert args.scheduler_cls == "vllm.v1.core.sched.scheduler.Scheduler"


@pytest.fixture(scope="module")
def engine_args_patched():
    import agentcache.entrypoints.cli.main  # noqa: F401 - triggers EngineArgs patch
    yield
