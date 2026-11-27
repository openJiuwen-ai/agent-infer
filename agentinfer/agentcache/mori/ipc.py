# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Explicit protocol over the vLLM core client's utility channel."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from functools import cache
from typing import get_type_hints

ATTEMPT_ARG = "agentinfer_mori_attempt"
PROGRAM_ARG = "mori_program_id"
CONTEXT_CHARS_ARG = "agentinfer_mori_context_chars"
PREFIX_ARG = "agentinfer_mori_prefix_key"
CHILD_COUNT_ARG = "agentinfer_mori_child_count"
PREPARATION_NS_ARG = "agentinfer_mori_preparation_ns"
RESERVED_ARGS = frozenset(
    (ATTEMPT_ARG, PROGRAM_ARG, CONTEXT_CHARS_ARG, PREFIX_ARG, CHILD_COUNT_ARG, PREPARATION_NS_ARG)
)
UTILITY_METHOD = "agentinfer_program_scheduling"


class Event(str, Enum):
    PROGRESS = "progress"
    FINISH = "finish"
    CANCEL = "cancel"
    RELEASE = "release"


def chat_facts(payload: dict) -> tuple[int, str]:
    """Port ThunderAgent router._prefix_group_key and its pre-template JSON size.

    Extract the original Chat facts before template rendering and tokenization.
    """
    parts = []
    for message in payload.get("messages") or ():
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                block["text"] for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)
            )
    tools = payload.get("tools")
    if tools is None and isinstance(payload.get("extra_body"), dict):
        tools = payload["extra_body"].get("tools")
    system = "\x00".join(parts)[:1024]
    tool_json = json.dumps(tools, sort_keys=True, ensure_ascii=False) if tools else ""
    key = "_empty_"
    if system or tool_json:
        key = hashlib.sha1((system + "\x01" + tool_json).encode("utf-8", "ignore"), usedforsecurity=False).hexdigest()[
            :16
        ]
    return len(json.dumps(payload, ensure_ascii=False)), key


def handle_event(admission, name: str, args: list):
    event = Event(name)
    if event is Event.PROGRESS:
        return admission.progress(*args)
    if event is Event.FINISH:
        return admission.finish(*args)
    if event is Event.CANCEL:
        return admission.cancel(*args)
    return admission.release(*args)


@cache
def _finish_returns_requests() -> bool:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

    # The Core pause/shutdown consumer uses the scheduler's native finish ABI.
    return get_type_hints(Scheduler.finish_requests)["return"] == list[Request]


def native_finished_requests(requests: list) -> list:
    """Return retained aborts in the same format as vLLM Scheduler.finish_requests."""
    if not requests or _finish_returns_requests():
        return requests
    return [(request.request_id, request.client_index) for request in requests]


def send_retained_aborts(core, admission) -> None:
    requests = admission.take_aborted_requests()
    if requests:
        core._send_abort_outputs(native_finished_requests(requests))


def install_core_utility() -> None:
    """Install the one entry point used by the opt-in EngineCore adapter."""
    from vllm.v1.engine.core import EngineCore

    def handle(self, name: str, args: list):
        admission = self.scheduler._agentcache.program_admission
        if admission is None:
            raise RuntimeError("EngineCore has no Program scheduling host")
        result = handle_event(admission, name, args)
        if name in (Event.CANCEL, Event.RELEASE, Event.FINISH):
            send_retained_aborts(self, admission)
        return result

    setattr(EngineCore, UTILITY_METHOD, handle)
