# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Expose AgentInfer components and optional vLLM runtime integration."""

AGENT_AWARE_SCHEDULER = "agentinfer.agentcache.core.scheduler.AgentAwareScheduler"

try:
    from vllm.engine.arg_utils import EngineArgs
except ModuleNotFoundError as exc:
    if exc.name != "vllm":
        raise
else:
    if not getattr(EngineArgs.__post_init__, "_agentinfer_patched", False):
        _original_post_init = EngineArgs.__post_init__

        def _patched_post_init(self):
            _original_post_init(self)
            if self.scheduler_cls is None:
                self.scheduler_cls = AGENT_AWARE_SCHEDULER

        _patched_post_init._agentinfer_patched = True
        EngineArgs.__post_init__ = _patched_post_init

    from agentinfer.agentcache.llm import LLM as LLM
