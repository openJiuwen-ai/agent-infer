# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Select the MORI Program scheduler, request scheduler and two-tier cache for vLLM."""

from __future__ import annotations

import math
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import replace

from agentinfer.agentcache.entrypoints.cli.serve_profile import (
    AgentInferServeError,
    ServeProfile,
    _coerce_additional_config,
)

_MORI_BRIDGE = "agentinfer.agentcache.mori.bridge"
_MORI_CONNECTOR = "agentinfer.agentcache.mori.connector"
_MORI_MIDDLEWARE = "agentinfer.agentcache.mori.admission_api.MoriSchedulingMiddleware"


def _settings(namespace: Namespace) -> tuple[dict, dict]:
    config = _coerce_additional_config(namespace.additional_config)
    agentcache = config.get("agentcache", {})
    if not isinstance(agentcache, dict) or not isinstance(agentcache.get("mori", {}), dict):
        raise AgentInferServeError("agentcache and agentcache.mori must be JSON objects")
    if agentcache.get("strategy", "mori") != "mori" or "progress_ttl" in agentcache:
        raise AgentInferServeError("agentcache config conflicts with --agentinfer-strategy mori")
    return agentcache, agentcache.get("mori", {})


@contextmanager
def embedded_runtime():
    from agentinfer.agentcache.mori.api import mori_parsed_request_body
    from agentinfer.agentcache.mori.vllm_compat import MoriVllmABIError, mori_text_detokenizer, require_mori_engine_abi

    try:
        require_mori_engine_abi()
    except MoriVllmABIError as exc:
        raise AgentInferServeError(str(exc)) from exc
    with mori_parsed_request_body(), mori_text_detokenizer():
        yield


def _configure_kv(namespace: Namespace) -> None:
    namespace.enable_prefix_caching = True
    namespace.enable_prompt_tokens_details = True

    transfer = getattr(namespace, "kv_transfer_config", None)
    if (
        transfer is None
        or transfer.kv_connector not in ("SimpleCPUOffloadConnector", "MoriCPUOffloadConnector")
        or transfer.kv_role != "kv_both"
        or transfer.kv_connector_extra_config.get("kv_offload_backend", "cpu") != "cpu"
    ):
        raise AgentInferServeError("MORI requires a RAM CPU offload --kv-transfer-config with kv_role=kv_both")
    namespace.kv_transfer_config = replace(
        transfer, kv_connector="MoriCPUOffloadConnector", kv_connector_module_path=_MORI_CONNECTOR
    )


def configure_embedded(namespace: Namespace) -> ServeProfile:
    agentcache, settings = _settings(namespace)
    supported = {
        "interval_seconds",
        "scheduler_interval",
        "joint_flashinfer_sampling",
        "sglang_unseeded_chat",
        "sglang_text_chat",
        "max_prefill_tokens",
    }
    if unknown := set(settings) - supported:
        raise AgentInferServeError(f"Unknown agentcache.mori settings: {sorted(unknown)}")
    for key in ("joint_flashinfer_sampling", "sglang_unseeded_chat", "sglang_text_chat"):
        if key in settings and type(settings[key]) is not bool:
            raise AgentInferServeError(f"agentcache.mori.{key} must be a boolean")
    interval = settings.get("interval_seconds", settings.get("scheduler_interval", 5))
    if type(interval) not in (float, int) or not math.isfinite(interval) or interval <= 0:
        raise AgentInferServeError("agentcache.mori.interval_seconds must be positive and finite")
    _configure_kv(namespace)
    # vLLM exposes this parsed Namespace to its API workers as app.state.args.
    namespace.agentinfer_mori_chat_mode = next(
        (key for key in ("sglang_unseeded_chat", "sglang_text_chat") if settings.get(key)), None
    )
    if "joint_flashinfer_sampling" in settings or settings.get("sglang_unseeded_chat"):
        agentcache = {**agentcache, "mori_joint_flashinfer_sampling": settings.get("joint_flashinfer_sampling", True)}
    if "max_prefill_tokens" in settings:
        agentcache = {**agentcache, "mori_max_prefill_tokens": settings["max_prefill_tokens"]}
    # vLLM installs --middleware entries in order; the last wraps the response wire.
    if _MORI_MIDDLEWARE in (getattr(namespace, "middleware", None) or ()):
        namespace.middleware = [name for name in namespace.middleware if name != _MORI_MIDDLEWARE]
    # The Program scheduler consumes only the scheduling interval.
    agentcache = {
        **agentcache,
        "strategy": "mori",
        "mori": {"interval_seconds": interval},
    }
    config = _coerce_additional_config(namespace.additional_config)
    namespace.additional_config = {**config, "agentcache": agentcache}
    return ServeProfile(
        async_scheduler_cls=f"{_MORI_BRIDGE}.MoriAsyncSchedulerBridge",
        sync_scheduler_cls=f"{_MORI_BRIDGE}.MoriSyncSchedulerBridge",
        middlewares=(_MORI_MIDDLEWARE,),
        agentcache_config=agentcache,
    )
