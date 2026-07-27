# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""Connect AgentInfer logs to an existing vLLM logging configuration."""

from __future__ import annotations

import logging


def attach_agentinfer_to_vllm_logging() -> None:
    """Reuse vLLM handlers without replacing vLLM, Uvicorn, or user logging.

    vLLM configures only its own logger hierarchy. AgentInfer uses a separate
    hierarchy, so its INFO records otherwise have no configured handler. The
    integration shares the already-created vLLM handlers and effective level;
    handler formatters, filters, streams, and color behavior remain native.

    Explicit AgentInfer handlers take precedence. Calling this function more
    than once is therefore safe and does not duplicate records.
    """
    agentinfer_logger = logging.getLogger("agentinfer")
    if agentinfer_logger.handlers:
        return

    vllm_logger = logging.getLogger("vllm")
    if not vllm_logger.handlers:
        return

    for handler in vllm_logger.handlers:
        agentinfer_logger.addHandler(handler)
    agentinfer_logger.setLevel(vllm_logger.getEffectiveLevel())
    agentinfer_logger.propagate = False
