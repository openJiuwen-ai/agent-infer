from agentcache.llm import LLM  # noqa: E402, F401 - after patch
from vllm.engine.arg_utils import EngineArgs

_original_post_init = EngineArgs.__post_init__


def _patched_post_init(self):
    _original_post_init(self)
    if self.scheduler_cls is None:
        self.scheduler_cls = "agentcache.core.scheduler.AgentScheduler"


EngineArgs.__post_init__ = _patched_post_init
