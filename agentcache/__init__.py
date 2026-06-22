"""Importing agentcache patches vLLM EngineArgs to default to AgentAwareScheduler."""

from vllm.engine.arg_utils import EngineArgs

AGENT_AWARE_SCHEDULER = "agentcache.core.scheduler.AgentAwareScheduler"

if not getattr(EngineArgs.__post_init__, "_agentcache_patched", False):
    _original_post_init = EngineArgs.__post_init__

    def _patched_post_init(self):
        _original_post_init(self)
        if self.scheduler_cls is None:
            self.scheduler_cls = AGENT_AWARE_SCHEDULER

    _patched_post_init._agentcache_patched = True
    EngineArgs.__post_init__ = _patched_post_init

from agentcache.llm import LLM  # noqa: E402, F401 - after patch
