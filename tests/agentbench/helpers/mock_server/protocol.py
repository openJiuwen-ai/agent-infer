# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Explicit legacy fixtures and self-consistent character-token responses."""

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Gate(StrictModel):
    id: str = Field(min_length=1)
    phase: Literal["before_first_chunk", "before_terminal"] = "before_terminal"


class Rule(StrictModel):
    request_id: str = Field(min_length=1)
    endpoint: str | None = None
    gate: Gate | None = None
    delay_ms: int = Field(0, ge=0)
    fault: (
        Literal["http_error", "omit_done", "truncate_stream", "usage_mismatch", "token_ids_missing", "token_ids_short"]
        | None
    ) = None


class Scenario(StrictModel):
    profile: Literal["run_legacy", "replay"]
    model: str = "replay-ci-model"
    max_model_len: int = Field(8192, gt=0)
    round_index: Literal[0, 1] = 0
    response_seed: int = 9
    wait_for_arm: bool = False
    first_chunk_delay_ms: int = Field(5, ge=0)
    chunk_interval_ms: int = Field(2, ge=0)
    chunk_tokens: int = Field(1, gt=0)
    rules: list[Rule] = Field(default_factory=list)


def tokens(body: dict) -> list[int]:
    """Count actual supported text; reject unsupported shapes instead of dropping them."""
    if body.get("tools") or body.get("chat_template_kwargs"):
        raise ValueError("tools/templates are outside the mock HTTP contract")
    prompt = body.get("prompt")
    if isinstance(prompt, list):
        decode(prompt)
        return prompt
    if isinstance(prompt, str):
        return list(map(ord, prompt))
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("expected prompt or nonempty messages")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("only plain-text chat messages are supported")
    return list(map(ord, "".join(message["content"] for message in messages)))


def decode(values: list[int]) -> str:
    if not isinstance(values, list) or any(
        type(value) is not int or not 0 <= value <= 0x10FFFF or 0xD800 <= value <= 0xDFFF for value in values
    ):
        raise ValueError("invalid Unicode token IDs")
    return "".join(map(chr, values))


def answer(scenario: Scenario, request_id: str, count: int) -> str:
    """Disjoint printable CJK alphabets guarantee different answers across rounds."""
    seed = f"{scenario.response_seed}:{request_id}".encode()
    raw = hashlib.shake_256(seed).digest(count * 2)
    base = 0x4E00 if scenario.round_index == 0 else 0x7000
    return "".join(chr(base + int.from_bytes(raw[i : i + 2], "big") % 8192) for i in range(0, len(raw), 2))
