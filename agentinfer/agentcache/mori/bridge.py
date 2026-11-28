# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Install Program admission on the synchronous or asynchronous MORI engine."""

from time import monotonic

from vllm.v1.request import RequestStatus

from agentinfer.agentcache.core.scheduler import add_to_native_waiting, admission_settings
from agentinfer.agentcache.core.vllm_logging import attach_agentinfer_to_vllm_logging
from agentinfer.agentcache.mori.admission import MoriProgramAdmissionController
from agentinfer.agentcache.mori.ipc import install_core_utility, native_finished_requests
from agentinfer.agentcache.mori.program_cache import HiRadixProgramCache
from agentinfer.agentcache.mori.scheduler import MoriAsyncScheduler, MoriScheduler
from agentinfer.agentcache.mori.timer import install_mori_timer


class _MoriProgramAdmissionMixin:
    def __init__(self, vllm_config, kv_cache_config, *args, **kwargs):
        settings, backend_id, _, _ = admission_settings(vllm_config)
        if settings.get("strategy") != "mori":
            raise ValueError("The MORI scheduler bridge requires the MORI Program strategy")
        super().__init__(vllm_config, kv_cache_config, *args, **kwargs)
        self.program_admission = None
        try:
            attach_agentinfer_to_vllm_logging()
            tree = self.tree_cache
            self.program_admission = MoriProgramAdmissionController.from_settings(
                HiRadixProgramCache(tree), settings, backend_id
            )
            tree.placement.current_ref = self.program_admission.runtime.registry.current_ref
            install_core_utility()
            install_mori_timer()
        except BaseException:
            self.shutdown()
            raise

    def add_request(self, request):
        queued_at = monotonic()
        native_add = super().add_request
        self.program_admission.add_request(request, lambda child: add_to_native_waiting(child, queued_at, native_add))

    def finish_requests(self, request_ids, finished_status):
        ids = request_ids if request_ids is None or isinstance(request_ids, str) else tuple(request_ids)
        retained = self.program_admission.finish_requests(ids, finished_status)
        return native_finished_requests(retained) + super().finish_requests(ids, finished_status)

    def shutdown(self):
        try:
            if self.program_admission is not None:
                self.program_admission.finish_requests(None, RequestStatus.FINISHED_ABORTED)
        finally:
            if self.program_admission is not None:
                self.tree_cache.placement.current_ref = None
                self.program_admission = None
            super().shutdown()


class MoriSyncSchedulerBridge(_MoriProgramAdmissionMixin, MoriScheduler):
    """Admit Programs and run synchronous MORI request batches."""


class MoriAsyncSchedulerBridge(_MoriProgramAdmissionMixin, MoriAsyncScheduler):
    """Admit Programs and run MORI batches with native async output bookkeeping."""
