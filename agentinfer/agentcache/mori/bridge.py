# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Install Program admission on the synchronous or asynchronous MORI engine."""

from agentinfer.agentcache.core.scheduler import _VllmAdmissionHooks
from agentinfer.agentcache.mori.program_cache import HiRadixProgramCache
from agentinfer.agentcache.mori.scheduler import MoriAsyncScheduler, MoriScheduler


class _MoriProgramAdmissionMixin:
    def __init__(self, vllm_config, kv_cache_config, *args, **kwargs):
        if vllm_config.additional_config.get("agentcache", {}).get("strategy") != "mori":
            raise ValueError("The MORI scheduler bridge requires the MORI Program strategy")
        super().__init__(vllm_config, kv_cache_config, *args, **kwargs)
        self._agentcache = self.program_cache_adapter = None
        try:
            self.program_cache_adapter = HiRadixProgramCache(self.tree_cache)
            self._agentcache = _VllmAdmissionHooks(self, vllm_config, kv_cache_config, self.block_size)
            self.tree_cache.placement.current_ref = self._agentcache.program_scheduling.runtime.registry.current_ref
        except BaseException:
            self.shutdown()
            raise

    def add_request(self, request):
        self._agentcache.add_request(self, request, super().add_request)

    def finish_requests(self, request_ids, finished_status):
        return self._agentcache.finish_requests(self, request_ids, finished_status, super().finish_requests)

    def shutdown(self):
        try:
            if self._agentcache is not None:
                self._agentcache.program_scheduling.cancel_gates()
        finally:
            if self.program_cache_adapter is not None:
                self.program_cache_adapter.tree.placement.current_ref = None
            self._agentcache = self.program_cache_adapter = None
            super().shutdown()


class MoriSyncSchedulerBridge(_MoriProgramAdmissionMixin, MoriScheduler):
    """Admit Programs and run synchronous MORI request batches."""


class MoriAsyncSchedulerBridge(_MoriProgramAdmissionMixin, MoriAsyncScheduler):
    """Admit Programs and run MORI batches with native async output bookkeeping."""
