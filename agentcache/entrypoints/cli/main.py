from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.cli.main import main as _vllm_main

_original_post_init = EngineArgs.__post_init__


def _patched_post_init(self):
    _original_post_init(self)
    if self.scheduler_cls is None:
        self.scheduler_cls = "agentcache.core.scheduler.AgentScheduler"


EngineArgs.__post_init__ = _patched_post_init


def main():
    _vllm_main()
