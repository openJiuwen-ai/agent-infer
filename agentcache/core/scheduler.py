from vllm.v1.core.sched.scheduler import Scheduler as _VllmScheduler

from agentcache.core.request_queue import AgentAwareQueue


class AgentAwareScheduler(_VllmScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waiting = AgentAwareQueue()
