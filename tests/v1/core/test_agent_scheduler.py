# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project

import pytest

from unittest.mock import MagicMock, patch

from agentcache.core.request_queue import AgentAwareQueue
from agentcache.core.scheduler import AgentAwareScheduler
from vllm.config.scheduler import SchedulerConfig
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.scheduler import Scheduler as VllmScheduler

pytestmark = pytest.mark.cpu_test


def test_agent_scheduler_is_subclass_of_vllm_scheduler():
    assert issubclass(AgentAwareScheduler, VllmScheduler)


def test_agent_scheduler_is_subclass_of_scheduler_interface():
    assert issubclass(AgentAwareScheduler, SchedulerInterface)


def test_get_scheduler_cls_resolves_agent_scheduler():
    cfg = SchedulerConfig.default_factory(
        max_model_len=8192,
        scheduler_cls="agentcache.core.scheduler.AgentAwareScheduler",
    )
    cls = cfg.get_scheduler_cls()
    assert cls is AgentAwareScheduler


def test_get_scheduler_cls_default_is_vllm_scheduler():
    cfg = SchedulerConfig.default_factory(max_model_len=8192)
    cls = cfg.get_scheduler_cls()
    assert cls is VllmScheduler


def test_patch_sets_scheduler_cls_when_none():
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(model="test-model")
    assert args.scheduler_cls == "agentcache.core.scheduler.AgentAwareScheduler"


def test_patch_preserves_explicit_scheduler_cls():
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(
        model="test-model",
        scheduler_cls="vllm.v1.core.sched.scheduler.Scheduler",
    )
    assert args.scheduler_cls == "vllm.v1.core.sched.scheduler.Scheduler"


def test_agent_scheduler_uses_agent_aware_queue():
    with patch.object(VllmScheduler, "__init__", lambda self: None):
        s = AgentAwareScheduler.__new__(AgentAwareScheduler)
        s.__init__()
        assert isinstance(s.waiting, AgentAwareQueue)


if __name__ == "__main__":
    pytest.main([__file__])
