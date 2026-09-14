# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""SSE usage observers for endpoint-specific streaming protocols.

Each observer parses server-sent events for its API style (Anthropic or
OpenAI-compatible) and extracts token-usage fields. Observer classes are
referenced from the agent runtime registry so that adding a new runtime
automatically wires the correct usage observer.
"""

import json
from collections.abc import Mapping


def _normalize_usage(usage: Mapping[str, object]) -> dict[str, int]:
    """Normalise vendor-specific usage-key names into canonical counters."""
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "cache_creation_input_tokens": ("cache_creation_input_tokens",),
        "cache_read_input_tokens": ("cache_read_input_tokens", "cached_tokens"),
    }
    normalized: dict[str, int] = {}
    for target, names in aliases.items():
        value = next((usage.get(name) for name in names if isinstance(usage.get(name), int)), None)
        if isinstance(value, int):
            normalized[target] = value
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict) and isinstance(prompt_details.get("cached_tokens"), int):
        normalized["cache_read_input_tokens"] = prompt_details["cached_tokens"]
    return normalized


class _AnthropicSSEUsageObserver:
    """Parse Anthropic SSE stream events and collect ``usage`` fields."""

    def __init__(self) -> None:
        self._buffer = b""
        self.usage: dict[str, int] = {}

    def observe(self, chunk: bytes) -> bool:
        self._buffer += chunk
        observed_event = False
        while True:
            lf_boundary = self._buffer.find(b"\n\n")
            crlf_boundary = self._buffer.find(b"\r\n\r\n")
            boundaries = [(index, size) for index, size in ((lf_boundary, 2), (crlf_boundary, 4)) if index >= 0]
            if not boundaries:
                break
            index, size = min(boundaries)
            event, self._buffer = self._buffer[:index], self._buffer[index + size :]
            data_lines = [line[5:].lstrip() for line in event.splitlines() if line.startswith(b"data:")]
            if not data_lines:
                continue
            try:
                payload = json.loads(b"\n".join(data_lines))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("type") == "content_block_delta":
                observed_event = True
            usage = payload.get("usage") if isinstance(payload, dict) else None
            if not isinstance(usage, dict) and isinstance(payload, dict):
                message = payload.get("message")
                usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict):
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ):
                    value = usage.get(key)
                    if isinstance(value, int):
                        self.usage[key] = value
        return observed_event


class _OpenAISSEUsageObserver:
    """Parse OpenAI-compatible SSE stream events and collect ``usage`` fields."""

    def __init__(self) -> None:
        self._buffer = b""
        self.usage: dict[str, int] = {}

    def observe(self, chunk: bytes) -> bool:
        self._buffer += chunk
        observed_content = False
        while True:
            lf_boundary = self._buffer.find(b"\n\n")
            crlf_boundary = self._buffer.find(b"\r\n\r\n")
            boundaries = [(index, size) for index, size in ((lf_boundary, 2), (crlf_boundary, 4)) if index >= 0]
            if not boundaries:
                break
            index, size = min(boundaries)
            event, self._buffer = self._buffer[:index], self._buffer[index + size :]
            data_lines = [line[5:].lstrip() for line in event.splitlines() if line.startswith(b"data:")]
            if not data_lines or data_lines == [b"[DONE]"]:
                continue
            try:
                payload = json.loads(b"\n".join(data_lines))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            choices = payload.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    if isinstance(delta, dict) and delta.get("content"):
                        observed_content = True
                        break
            usage = payload.get("usage")
            if isinstance(usage, dict):
                self.usage.update(_normalize_usage(usage))
        return observed_content
