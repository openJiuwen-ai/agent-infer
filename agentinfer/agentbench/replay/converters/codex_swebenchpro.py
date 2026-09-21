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
from ..local_tokenizer import LocalTokenizerCounter, discover_local_tokenizer
from ..tokenizer_retry import TOKENIZER_REQUEST_MAX_ATTEMPTS, TOKENIZER_RETRY_BACKOFF_SECONDS
from ..unified_trace_ir import write_trace_ir_manifest
from .base import ConverterSummary, ReplayDatasetConverter

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

    def input_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count one complete conversation using the Backend chat template."""

        ...


class _BackendTokenizer:
    """Synchronous adapter for the configured Backend ``/tokenize`` endpoint."""

    def __init__(self, config: ReplayBenchConfig) -> None:
        self.model = config.backend.model
        self.base_url = config.backend.resolved_tokenizer_base_url.rstrip("/")
        self.chat_template_kwargs = dict(config.backend.chat_template_kwargs)
        headers = {}
        if config.backend.api_key_env:
            headers["authorization"] = f"Bearer {os.environ[config.backend.api_key_env]}"
        self.client = httpx.Client(
            timeout=config.replay.request_timeout_seconds,
            headers=headers,
            trust_env=False,
        )
        self._count_cache: dict[bytes, int] = {}
        self.tokenizer_operations = 0
        try:
            self.local_tokenizer: LocalTokenizerCounter | None = discover_local_tokenizer(
                config,
                get_json=self._get_json,
                backend_tokens=self._request_tokens,
            )
        except Exception:
            self.client.close()
            raise

    def close(self) -> None:
        """Close the synchronous Backend HTTP client."""

        self.client.close()

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        """Tokenize one payload, retrying transport failures but not invalid responses."""

        self.tokenizer_operations += 1
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
        if not isinstance(response_body, dict):
            raise ValueError("Backend /tokenize response must be an object")
        return response_body

    def _request_count(self, payload: dict[str, object]) -> int:
        response_body = self._request(payload)
        count = response_body.get("count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
        tokens = response_body.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
        raise ValueError("Backend /tokenize response contains neither count nor tokens")

    def _request_tokens(self, payload: dict[str, object]) -> tuple[int, ...] | None:
        tokens = self._request(payload).get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            return None
        return tuple(tokens)

    def _get_json(self, url: str) -> dict[str, object] | None:
        try:
            response = self.client.get(url)
            if response.status_code != 200:
                return None
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except (AttributeError, httpx.HTTPError, ValueError):
            return None

    def _count(self, text: str) -> int:
        cache_key = hashlib.sha256(text.encode()).digest()
        count = self._count_cache.get(cache_key)
        if count is None:
            count = self._request_count({"prompt": text, "add_special_tokens": False})
            self._count_cache[cache_key] = count
        return count

    def content_tokens(self, text: str) -> int:
        """Count raw content locally when validated, otherwise through the Backend."""

        if self.local_tokenizer is not None:
            return self.local_tokenizer.content_tokens(text)
        return self._count(text)

    def input_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count a conversation with the configured model's validated chat template."""

        if self.local_tokenizer is not None:
            return self.local_tokenizer.input_tokens(list(messages))
        payload: dict[str, object] = {
            "messages": list(messages),
            "add_generation_prompt": True,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        return self._request_count(payload)

    @property
    def counting_mode(self) -> str:
        if self.local_tokenizer is None:
            return "backend"
        return "local_incremental" if self.local_tokenizer.incremental else "local_full_chat"


class CodexSwebenchProConverter(ReplayDatasetConverter):
    """Convert plain-text human/gpt turns into single-agent serial Trace IR.

    Source timestamps are unavailable; generated timestamps encode turn order
    only. Replay config requires trace-record prompts and Lognormal intervals.
    """

    name = "codex_swebenchpro"

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
                messages: list[dict[str, str]] = []
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
                    messages.append({"role": "user", "content": human_text})
                    input_tokens = self.tokenizer.input_tokens(messages)
                    output_tokens = self.tokenizer.content_tokens(assistant_text)
                    if input_tokens <= 0 or output_tokens <= 0:
                        raise ValueError(f"{session_id} turn {turn_index} produced a non-positive token count")
                    messages.append({"role": "assistant", "content": assistant_text})
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
            "tokenizer_counting_mode": getattr(self.tokenizer, "counting_mode", "custom"),
            "tokenizer_operations": getattr(self.tokenizer, "tokenizer_operations", None),
        }
        write_trace_ir_manifest(
            output_dir,
            converter_name=self.name,
            source_path=source,
            summary=summary_payload,
        )
        return summary
