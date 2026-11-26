# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Explicit protocol over the vLLM core client's utility channel."""

from __future__ import annotations

import hashlib
import json
from enum import Enum

UTILITY_METHOD = "agentinfer_program_scheduling"


class Event(str, Enum):
    BEGIN = "begin"
    PROGRESS = "progress"
    FINISH = "finish"
    CANCEL = "cancel"
    RELEASE = "release"


def chat_facts(payload: dict) -> tuple[int, str]:
    """Port ThunderAgent router._prefix_group_key and its pre-template JSON size.

    Extract the facts in the API process so the Core can receive a compact
    admission message ahead of vLLM's native request.
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
    if event is Event.BEGIN:
        return admission.begin(*args)
    if event is Event.PROGRESS:
        return admission.progress(*args)
    if event is Event.FINISH:
        return admission.finish(*args)
    if event is Event.CANCEL:
        return admission.cancel(*args)
    return admission.release(*args)


def install_core_utility() -> None:
    """Install the one entry point used by the opt-in EngineCore adapter."""
    from vllm.v1.engine.core import EngineCore

    def handle(self, name: str, args: list):
        admission = self.scheduler._agentcache.program_admission
        if admission is None:
            raise RuntimeError("EngineCore has no Program scheduling host")
        return handle_event(admission, name, args)

    setattr(EngineCore, UTILITY_METHOD, handle)
