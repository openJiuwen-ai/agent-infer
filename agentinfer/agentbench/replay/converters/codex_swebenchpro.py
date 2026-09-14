# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Convert Inferact conversations during Replay input preparation.

``CodexSwebenchProConverter`` owns source parsing and unified Trace IR emission. Token
accounting is delegated to the configured Backend; planning and execution do
not depend on this dataset-specific module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import httpx

from ..config import ReplayBenchConfig
from ..tokenizer_retry import TOKENIZER_REQUEST_MAX_ATTEMPTS, TOKENIZER_RETRY_BACKOFF_SECONDS
from ..unified_trace_ir import write_trace_ir_manifest
from .base import ConverterSummary, ReplayDatasetConverter

_CONVERTER_VERSION = "agentinfer-codex-swebenchpro/v3"
logger = logging.getLogger(__name__)


def _iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, object]]:
    """Stream a top-level JSON array, while accepting one inspect-sized object."""

    with path.open(encoding="utf-8") as probe:
        first = next((character for character in iter(lambda: probe.read(1), "") if not character.isspace()), "")
    if first == "{":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        yield value
        return

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False
    expect_value = True
    after_comma = False
    with path.open(encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            eof = not chunk
            buffer = buffer[position:] + chunk
            position = 0
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError("Codex source must be a top-level JSON array")
                    position += 1
                    started = True
                    continue
                if position >= len(buffer):
                    break
                if expect_value:
                    if buffer[position] == "]":
                        if after_comma:
                            raise ValueError("Codex source array must not contain a trailing comma")
                        position += 1
                        finished = True
                        break
                    try:
                        value, end = decoder.raw_decode(buffer, position)
                    except json.JSONDecodeError as exc:
                        if eof:
                            raise ValueError("truncated or invalid Codex source JSON") from exc
                        break
                    if not isinstance(value, dict):
                        raise ValueError("every Codex source array item must be an object")
                    yield value
                    position = end
                    expect_value = False
                    after_comma = False
                    continue
                if buffer[position] == ",":
                    position += 1
                    expect_value = True
                    after_comma = True
                    continue
                if buffer[position] == "]":
                    position += 1
                    finished = True
                    break
                raise ValueError("Codex source array items must be separated by one comma")
            if finished:
                if buffer[position:].strip() or handle.read().strip():
                    raise ValueError("unexpected data after Codex source array")
                return
            if eof:
                raise ValueError("Codex source array has no closing bracket")


class _TraceTokenizer(Protocol):
    """Token-accounting contract consumed by the Inferact converter."""

    def close(self) -> None:
        """Release tokenizer resources after conversion."""

        ...

    def content_tokens(self, text: str) -> int:
        """Count content tokens without chat-template framing."""

        ...

    def trace_turn_tokens(self, completed_tokens: int, human: str, assistant: str) -> tuple[int, int]:
        """Return current input tokens and the updated completed transcript count."""

        ...


class _BackendTokenizer:
    """Synchronous adapter for the configured Backend ``/tokenize`` endpoint."""

    def __init__(self, config: ReplayBenchConfig) -> None:
        self.model = config.backend.model
        self.base_url = config.backend.resolved_tokenizer_base_url.rstrip("/")
        headers = {}
        if config.backend.api_key_env:
            headers["authorization"] = f"Bearer {os.environ[config.backend.api_key_env]}"
        self.client = httpx.Client(
            timeout=config.replay.request_timeout_seconds,
            headers=headers,
            trust_env=False,
        )
        self._count_cache: dict[bytes, int] = {}
        try:
            self.generation_prefix_tokens = self._count("<|im_start|>assistant\n")
            self._validate_chat_template()
        except BaseException:
            self.client.close()
            raise

    def close(self) -> None:
        """Close the synchronous Backend HTTP client."""

        self.client.close()

    def _request_count(self, payload: dict[str, object]) -> int:
        """Count tokens, retrying transport failures but not invalid responses."""

        for attempt in range(1, TOKENIZER_REQUEST_MAX_ATTEMPTS + 1):
            try:
                response = self.client.post(
                    f"{self.base_url}/tokenize",
                    json={"model": self.model, **payload},
                )
                break
            except httpx.TransportError as exc:
                if attempt == TOKENIZER_REQUEST_MAX_ATTEMPTS:
                    raise
                delay = TOKENIZER_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Converter tokenizer transport failure; retrying: path=/tokenize attempt=%d/%d error=%s delay_seconds=%s",
                    attempt,
                    TOKENIZER_REQUEST_MAX_ATTEMPTS,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
        response.raise_for_status()
        response_body = response.json()
        count = response_body.get("count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
        tokens = response_body.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        raise ValueError("Backend /tokenize response contains neither count nor tokens")

    def _count(self, text: str) -> int:
        cache_key = hashlib.sha256(text.encode()).digest()
        count = self._count_cache.get(cache_key)
        if count is None:
            count = self._request_count({"prompt": text, "add_special_tokens": False})
            self._count_cache[cache_key] = count
        return count

    def _validate_chat_template(self) -> None:
        messages = [
            {"role": "user", "content": "agentinfer template probe"},
            {"role": "assistant", "content": "probe response"},
            {"role": "user", "content": "next probe"},
        ]
        complete_count = self._request_count(
            {
                "messages": messages,
                "add_generation_prompt": True,
            }
        )
        incremental_count = (
            self._wire_chunk_tokens("user", "agentinfer template probe")
            + self._wire_chunk_tokens("assistant", "probe response")
            + self._wire_chunk_tokens("user", "next probe")
            + self.generation_prefix_tokens
        )
        if complete_count != incremental_count:
            raise ValueError(
                "Backend tokenizer chat template is incompatible with the audited incremental Qwen ChatML layout"
            )

    def _wire_chunk_tokens(self, role: str, text: str) -> int:
        return self._count(f"<|im_start|>{role}\n{text}<|im_end|>\n")

    def content_tokens(self, text: str) -> int:
        """Count raw content through the Backend tokenizer endpoint."""

        return self._count(text)

    def trace_turn_tokens(self, completed_tokens: int, human: str, assistant: str) -> tuple[int, int]:
        """Increment the audited Qwen ChatML transcript using Backend counts."""

        user_tokens = self._wire_chunk_tokens("user", human)
        input_tokens = completed_tokens + user_tokens + self.generation_prefix_tokens
        completed_tokens += user_tokens + self._wire_chunk_tokens("assistant", assistant)
        return input_tokens, completed_tokens


class CodexSwebenchProConverter(ReplayDatasetConverter):
    """Convert plain-text human/gpt turns into single-agent serial Trace IR.

    Source timestamps are unavailable; generated timestamps encode turn order
    only. Replay config requires trace-record prompts and Lognormal intervals.
    """

    name = "codex_swebenchpro"
    version = _CONVERTER_VERSION

    def __init__(self, tokenizer: _TraceTokenizer) -> None:
        """Create a converter with an explicit token-accounting provider."""

        self.tokenizer = tokenizer

    @classmethod
    def from_backend(cls, config: ReplayBenchConfig) -> CodexSwebenchProConverter:
        """Create a runtime converter backed by the configured tokenizer service."""

        return cls(_BackendTokenizer(config))

    def close(self) -> None:
        """Release the tokenizer resources owned by this converter."""

        self.tokenizer.close()

    def convert(self, source: Path, output_dir: Path) -> ConverterSummary:
        """Write strict human/assistant pairs as Trace IR; callers validate before use."""

        source = source.resolve()
        output_dir = output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"converter output directory must be empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        text_root = output_dir / "texts"
        text_root.mkdir()
        requests_path = output_dir / "requests.jsonl"

        session_count = 0
        request_count = 0
        base_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
        with requests_path.open("w", encoding="utf-8") as requests_handle:
            for session_index, record in enumerate(_iter_json_array(source)):
                conversations = record.get("conversations")
                if not isinstance(conversations, list) or not conversations:
                    raise ValueError(f"session {session_index} has no conversations list")
                if len(conversations) % 2:
                    raise ValueError(f"session {session_index} has an odd number of conversation messages")
                session_id = f"codex-session-{session_index:04d}"
                session_dir = text_root / session_id
                session_dir.mkdir()
                completed_tokens = 0
                for turn_index in range(len(conversations) // 2):
                    human = conversations[turn_index * 2]
                    assistant = conversations[turn_index * 2 + 1]
                    if not isinstance(human, dict) or human.get("from") not in {"human", "user"}:
                        raise ValueError(f"{session_id} turn {turn_index} does not start with human content")
                    if not isinstance(assistant, dict) or assistant.get("from") not in {"gpt", "assistant"}:
                        raise ValueError(f"{session_id} turn {turn_index} has no assistant response")
                    human_text = human.get("value")
                    assistant_text = assistant.get("value")
                    if not isinstance(human_text, str) or not isinstance(assistant_text, str):
                        raise ValueError(f"{session_id} turn {turn_index} contains non-text content")

                    text_path = session_dir / f"turn_{turn_index}.txt"
                    text_path.write_text(human_text, encoding="utf-8")
                    input_tokens, completed_tokens = self.tokenizer.trace_turn_tokens(
                        completed_tokens,
                        human_text,
                        assistant_text,
                    )
                    output_tokens = self.tokenizer.content_tokens(assistant_text)
                    if input_tokens <= 0 or output_tokens <= 0:
                        raise ValueError(f"{session_id} turn {turn_index} produced a non-positive token count")
                    synthetic = base_time + timedelta(microseconds=turn_index)
                    row = {
                        "request_id": f"{session_id}-turn-{turn_index:04d}",
                        "session_id": session_id,
                        "task_id": session_id,
                        "actor_id": "lead",
                        "actor_role": "lead",
                        "started_at": synthetic.isoformat(),
                        "finished_at": synthetic.isoformat(),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_tokens": None,
                        "status": "success",
                        "request_purpose": "lead_main" if turn_index == 0 else "continuation",
                        "prompt_ref": {"session_id": session_id, "turn_index": turn_index},
                    }
                    requests_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    request_count += 1
                session_count += 1

        if session_count == 0 or request_count == 0:
            raise ValueError("Codex source contains no replayable conversation turns")
        summary = ConverterSummary(self.name, session_count, request_count, request_count)
        summary_payload = {
            **summary.to_dict(),
            "source_records_consumed": session_count,
        }
        write_trace_ir_manifest(
            output_dir,
            converter_name=self.name,
            converter_version=self.version,
            source_path=source,
            summary=summary_payload,
        )
        return summary
