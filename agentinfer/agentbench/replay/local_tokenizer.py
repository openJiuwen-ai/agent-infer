# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Discover and validate a Backend-compatible local tokenizer for Replay.

The module owns local tokenizer loading and Backend token-ID validation.
Dataset parsing, tokenizer ownership, and HTTP retry policy remain with callers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from .config import ReplayBenchConfig

logger = logging.getLogger(__name__)

_PROBE_TEXT = "AgentInfer tokenizer probe: ASCII, 中文, newline\nend"
_INCREMENTAL_MESSAGE_COUNTS = (1, 3, 5, 9)
_PROBE_MESSAGES = [
    {"role": "user", "content": "AgentInfer template probe\nline two"},
    {"role": "assistant", "content": "probe response"},
    {"role": "user", "content": "next probe 中文"},
]
_ADDITIVITY_VARIANTS = (
    ("plain user", "plain assistant", "next user"),
    (" leading user\n", "assistant trailing ", "中文与 punctuation: []{}"),
    ("<tag>user</tag>", "line one\nline two", "final"),
    ("thinking probe", "<think>hidden reasoning</think>visible answer", "after thinking"),
)


class _Encoding(Protocol):
    ids: list[int]


class _Tokenizer(Protocol):
    """Minimal Transformers tokenizer surface used by local Replay counting."""

    chat_template: object

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int] | _Encoding:
        """Encode text without implicit special tokens."""

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        """Decode token IDs while preserving special tokens."""

    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: Any) -> str:
        """Render messages using the model chat template."""


class LocalTokenizerCounter:
    """Count raw text and chat prompts without Backend HTTP round trips."""

    def __init__(self, tokenizer: _Tokenizer, chat_template_kwargs: dict[str, object]) -> None:
        self.tokenizer = tokenizer
        self.chat_template_kwargs = dict(chat_template_kwargs)
        self._content_cache: dict[str, tuple[int, ...]] = {}
        self._single_message_cache: dict[tuple[str, str], int] = {}
        self._offset_cache: dict[int, int] = {}
        self.incremental = self._supports_incremental_counting()

    def _encode(self, text: str) -> tuple[int, ...]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if hasattr(encoded, "ids"):
            encoded = encoded.ids
        return tuple(int(token) for token in encoded)

    def text_token_ids(self, text: str) -> tuple[int, ...]:
        tokens = self._content_cache.get(text)
        if tokens is None:
            tokens = self._encode(text)
            self._content_cache[text] = tokens
        return tokens

    def detokenize_tokens(self, tokens: tuple[int, ...] | list[int]) -> str:
        return str(self.tokenizer.decode(list(tokens), skip_special_tokens=False))

    def content_tokens(self, text: str) -> int:
        return len(self.text_token_ids(text))

    def chat_token_ids(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]] | None = None,
    ) -> tuple[int, ...]:
        kwargs = dict(self.chat_template_kwargs)
        kwargs.update(tokenize=False, add_generation_prompt=True)
        if tools:
            kwargs["tools"] = tools
        rendered = self.tokenizer.apply_chat_template(messages, **kwargs)
        if not isinstance(rendered, str):
            raise TypeError("local tokenizer chat template did not render text")
        return self._encode(rendered)

    def _single_message_count(self, role: str, content: str) -> int:
        key = (role, content)
        count = self._single_message_cache.get(key)
        if count is None:
            count = len(self.chat_token_ids([{"role": role, "content": content}], tools=None))
            self._single_message_cache[key] = count
        return count

    @staticmethod
    def _probe_history(message_count: int, variant: tuple[str, str, str]) -> list[dict[str, object]]:
        user, assistant, next_user = variant
        messages: list[dict[str, object]] = []
        for index in range(message_count):
            role = "user" if index % 2 == 0 else "assistant"
            if role == "assistant":
                content = f"{assistant} {index}"
            else:
                content = f"{user if index == 0 else next_user} {index}"
            messages.append({"role": role, "content": content})
        return messages

    def _offset(self, message_count: int) -> int:
        offset = self._offset_cache.get(message_count)
        if offset is None:
            messages = self._probe_history(message_count, _ADDITIVITY_VARIANTS[0])
            full = len(self.chat_token_ids(messages))
            singles = sum(
                self._single_message_count(str(message["role"]), str(message["content"])) for message in messages
            )
            offset = full - singles
            self._offset_cache[message_count] = offset
        return offset

    def _incremental_count(self, messages: list[dict[str, object]]) -> int:
        return sum(
            self._single_message_count(str(message["role"]), str(message["content"])) for message in messages
        ) + self._offset(len(messages))

    def _supports_incremental_counting(self) -> bool:
        """Accept additive counting only when varied multi-turn probes are exact."""

        try:
            for message_count in _INCREMENTAL_MESSAGE_COUNTS:
                expected_offset = self._offset(message_count)
                for variant in _ADDITIVITY_VARIANTS[1:]:
                    messages = self._probe_history(message_count, variant)
                    full = len(self.chat_token_ids(messages))
                    singles = sum(
                        self._single_message_count(str(message["role"]), str(message["content"]))
                        for message in messages
                    )
                    if full - singles != expected_offset:
                        return False
        except Exception:
            logger.debug("Local tokenizer incremental probe failed", exc_info=True)
            return False
        return True

    @staticmethod
    def _is_alternating_text_history(messages: list[dict[str, object]]) -> bool:
        return (
            bool(messages)
            and len(messages) % 2 == 1
            and all(
                message.get("role") == ("user" if index % 2 == 0 else "assistant")
                and isinstance(message.get("content"), str)
                for index, message in enumerate(messages)
            )
        )

    def input_tokens(self, messages: list[dict[str, object]]) -> int:
        # An offset computed for an unprobed length does not establish additivity.
        if (
            self.incremental
            and len(messages) in _INCREMENTAL_MESSAGE_COUNTS
            and self._is_alternating_text_history(messages)
        ):
            return self._incremental_count(messages)
        return len(self.chat_token_ids(messages))


def discover_local_tokenizer(
    config: ReplayBenchConfig,
    *,
    get_json: Callable[[str], dict[str, object] | None],
    backend_tokens: Callable[[dict[str, object]], tuple[int, ...] | None],
) -> LocalTokenizerCounter | None:
    """Find local tokenizer files and require exact Backend token-ID probes."""

    tokenizer_url = config.backend.resolved_tokenizer_base_url.rstrip("/")
    backend_url = config.backend.base_url.rstrip("/")
    tokenizer_info = get_json(f"{tokenizer_url}/tokenizer_info") or {}
    model_lists = [get_json(f"{tokenizer_url}/v1/models") or {}]
    if backend_url != tokenizer_url:
        model_lists.append(get_json(f"{backend_url}/v1/models") or {})
    candidates: list[str] = []
    for field in ("name_or_path", "_name_or_path"):
        value = tokenizer_info.get(field)
        if isinstance(value, str):
            candidates.append(value)
    for model_list in model_lists:
        data = model_list.get("data")
        if isinstance(data, list):
            matching = [item for item in data if isinstance(item, dict) and item.get("id") == config.backend.model]
            for item in matching or [item for item in data if isinstance(item, dict)]:
                root = item.get("root")
                if isinstance(root, str):
                    candidates.append(root)
    candidates.append(config.backend.model)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.info("Local tokenizer unavailable because transformers is not installed; using Backend /tokenize")
        return None

    template = tokenizer_info.get("chat_template")
    for candidate in dict.fromkeys(candidates):
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
            if isinstance(template, str) and template:
                tokenizer.chat_template = template
            counter = LocalTokenizerCounter(tokenizer, config.backend.chat_template_kwargs)
            raw_backend = backend_tokens({"prompt": _PROBE_TEXT, "add_special_tokens": False})
            chat_payload: dict[str, object] = {
                "messages": list(_PROBE_MESSAGES),
                "add_generation_prompt": True,
            }
            if config.backend.chat_template_kwargs:
                chat_payload["chat_template_kwargs"] = dict(config.backend.chat_template_kwargs)
            chat_backend = backend_tokens(chat_payload)
            if raw_backend is None or chat_backend is None:
                logger.info("Backend does not return token IDs; using Backend /tokenize for auditable counts")
                return None
            if counter.text_token_ids(_PROBE_TEXT) != raw_backend:
                logger.warning("Local tokenizer raw token IDs differ from Backend for candidate=%s", candidate)
                continue
            local_chat = counter.chat_token_ids(list(_PROBE_MESSAGES))
            if local_chat != chat_backend:
                logger.warning("Local tokenizer chat token IDs differ from Backend for candidate=%s", candidate)
                continue
            logger.info(
                "Using Backend-validated local tokenizer: candidate=%s counting_mode=%s",
                candidate,
                "incremental" if counter.incremental else "full_chat",
            )
            return counter
        except Exception:
            logger.debug("Cannot use local tokenizer candidate=%s", candidate, exc_info=True)
    logger.info("No Backend-compatible local tokenizer found; using Backend /tokenize")
    return None
