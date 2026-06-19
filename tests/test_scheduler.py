import unittest

from agentcache.core.scheduler import AgentScheduler
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.scheduler import Scheduler as VllmScheduler
from vllm.config.scheduler import SchedulerConfig


class TestAgentSchedulerResolution(unittest.TestCase):
    def test_agent_scheduler_is_subclass_of_vllm_scheduler(self):
        self.assertTrue(issubclass(AgentScheduler, VllmScheduler))

    def test_agent_scheduler_is_subclass_of_scheduler_interface(self):
        self.assertTrue(issubclass(AgentScheduler, SchedulerInterface))

    def test_get_scheduler_cls_resolves_agent_scheduler(self):
        cfg = SchedulerConfig.default_factory(
            max_model_len=8192,
            scheduler_cls="agentcache.core.scheduler.AgentScheduler",
        )
        cls = cfg.get_scheduler_cls()
        self.assertIs(cls, AgentScheduler)

    def test_get_scheduler_cls_default_is_vllm_scheduler(self):
        cfg = SchedulerConfig.default_factory(max_model_len=8192)
        cls = cfg.get_scheduler_cls()
        self.assertIs(cls, VllmScheduler)


class TestEngineArgsPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import agentcache.entrypoints.cli.main  # noqa: F401 - triggers EngineArgs patch

    def test_patch_sets_scheduler_cls_when_none(self):
        from vllm.engine.arg_utils import EngineArgs

        args = EngineArgs(model="test-model")
        self.assertEqual(args.scheduler_cls, "agentcache.core.scheduler.AgentScheduler")

    def test_patch_preserves_explicit_scheduler_cls(self):
        from vllm.engine.arg_utils import EngineArgs

        args = EngineArgs(
            model="test-model",
            scheduler_cls="vllm.v1.core.sched.scheduler.Scheduler",
        )
        self.assertEqual(
            args.scheduler_cls, "vllm.v1.core.sched.scheduler.Scheduler"
        )


if __name__ == "__main__":
    unittest.main()
