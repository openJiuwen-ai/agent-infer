import logging

from vllm.v1.core.sched.scheduler import Scheduler as _VllmScheduler

from agentinfer.agentcache.core.request_queue import AgentAwareQueue

logger = logging.getLogger(__name__)


class AgentAwareScheduler(_VllmScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: accept scheduler config when agent-aware policies need it.
        self.waiting = AgentAwareQueue()
        logger.warning("AgentAwareQueue initialized as waiting queue")
