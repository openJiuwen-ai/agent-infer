# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validated configuration for MORI relative-idleness placement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class MORIMode(str, Enum):
    """Supported MORI runtime mode."""

    FULL = "full"


@dataclass(frozen=True)
class MORIConfig:
    """Policy parameters whose values are independent of native cache capacity."""

    window_size: int = 5
    mode: MORIMode = MORIMode.FULL
    enable_wait_credit: bool = True
    ready_wait_limit_seconds: float = 0.0
    gpu_service_requests: int = 0
    history_gated_gpu_service: bool = False
    decode_batch_tokens: int = 0
    audit_writeback_sources: bool = False
    repair_writeback_endpoints: bool = False
    fast_capacity_accounting: bool = False
    decode_buffer_tokens: int = 100
    slow_request_threshold_seconds: float = 30.0
    slow_request_repeat_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("MORI window_size must be positive")
        if not isinstance(self.mode, MORIMode):
            raise ValueError("MORI mode must be a MORIMode")
        if not isinstance(self.enable_wait_credit, bool):
            raise ValueError("MORI enable_wait_credit must be a boolean")
        if not isinstance(self.audit_writeback_sources, bool):
            raise ValueError("MORI audit_writeback_sources must be a boolean")
        if not isinstance(self.repair_writeback_endpoints, bool):
            raise ValueError("MORI repair_writeback_endpoints must be a boolean")
        if not isinstance(self.fast_capacity_accounting, bool):
            raise ValueError("MORI fast_capacity_accounting must be a boolean")
        if (
            isinstance(self.gpu_service_requests, bool)
            or not isinstance(self.gpu_service_requests, int)
            or self.gpu_service_requests < 0
        ):
            raise ValueError("MORI gpu_service_requests must be a non-negative integer")
        if not isinstance(self.history_gated_gpu_service, bool):
            raise ValueError("MORI history_gated_gpu_service must be a boolean")
        if self.history_gated_gpu_service and self.gpu_service_requests <= 0:
            raise ValueError("MORI history_gated_gpu_service requires full mode and a positive service budget")
        if (
            isinstance(self.decode_batch_tokens, bool)
            or not isinstance(self.decode_batch_tokens, int)
            or self.decode_batch_tokens < 0
        ):
            raise ValueError("MORI decode_batch_tokens must be a non-negative integer")
        if (
            isinstance(self.ready_wait_limit_seconds, bool)
            or not math.isfinite(self.ready_wait_limit_seconds)
            or self.ready_wait_limit_seconds < 0
        ):
            raise ValueError("MORI ready_wait_limit_seconds must be finite and non-negative")
        if self.decode_buffer_tokens < 0:
            raise ValueError("MORI decode_buffer_tokens must be non-negative")
        if (
            isinstance(self.slow_request_threshold_seconds, bool)
            or not math.isfinite(self.slow_request_threshold_seconds)
            or self.slow_request_threshold_seconds < 0
        ):
            raise ValueError("MORI slow_request_threshold_seconds must be finite and non-negative")
        if (
            isinstance(self.slow_request_repeat_seconds, bool)
            or not math.isfinite(self.slow_request_repeat_seconds)
            or self.slow_request_repeat_seconds <= 0
        ):
            raise ValueError("MORI slow_request_repeat_seconds must be finite and positive")
