# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Read-only scheduler snapshots for paired MORI diagnostic runs."""

import json
import os
from array import array
from hashlib import sha256
from time import monotonic_ns

from vllm.logger import init_logger

logger = init_logger("vllm.agentinfer.mori.decisions")


class DecisionTrace:
    def __init__(self):
        self.request_keys = {}
        self.occurrences = {}

    @classmethod
    def from_environment(cls):
        return cls() if os.environ.get("AGENTINFER_MORI_DECISION_TRACE") == "1" else None

    def key(self, request):
        rid = request.request_id
        if rid not in self.request_keys:
            digest = sha256(array("i", request.prompt_token_ids or ())).hexdigest()
            occurrence = self.occurrences.get(digest, 0) + 1
            self.occurrences[digest] = occurrence
            self.request_keys[rid] = f"{digest}/{occurrence}"
        return self.request_keys[rid]

    def snapshot(self, scheduler):
        tree = scheduler.tree_cache

        def state(request):
            device, host = tree.request_match(request, include_loading=True, touch=False)
            return {
                "request": self.key(request),
                "computed": request.num_computed_tokens,
                "generated": request.num_output_tokens,
                "output_sha256": sha256(array("i", request.output_token_ids)).hexdigest(),
                "max_output": request.max_tokens,
                "hbm_hit": len(device) * tree.page_size,
                "dram_hit": len(host) * tree.page_size,
            }

        return {
            "waiting": [state(request) for request in scheduler.waiting],
            "holding": [state(request) for request in scheduler.kv_holding_waiting],
            "running": [state(request) for request in scheduler.running],
            "hbm_free": tree.device.free_uncached() * tree.page_size,
            "hbm_reclaimable": len(tree.device.ready) * tree.page_size,
            "dram_free": tree.host.free_uncached() * tree.page_size,
            "h2d_pages": len(tree._loading),
            "decode_ratio": round(scheduler.new_token_ratio, 9),
        }

    def emit(self, event, **fields):
        logger.info(
            "MORI_DECISION %s", json.dumps({"time_ns": monotonic_ns(), "event": event, **fields}, separators=(",", ":"))
        )
