# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Define converter and hash-block rendering contracts.

Dataset-specific source parsing lives in sibling modules. This module only owns
the reusable converter result and deterministic block-text interfaces.

BlockTokenizer and DeterministicBlockRenderer are forward scaffolding for future
block-hash converters. Only tests use them today; the Inferact converter emits
text-based IR. AgentX conversion and its full-prompt/turn mapping are not implemented.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol


@dataclass(frozen=True)
class ConverterSummary:
    """Minimum reviewer-facing coverage facts every converter must report."""

    dataset: str
    sessions: int
    requests: int
    text_files: int

    def to_dict(self) -> dict[str, object]:
        """Serialize converter coverage for the unified Trace IR manifest."""

        return asdict(self)


class ReplayDatasetConverter(ABC):
    """Convert one source dataset into the engine-neutral Replay Trace IR."""

    name: str
    version: str

    @abstractmethod
    def convert(self, source: Path, output_dir: Path) -> ConverterSummary:
        """Write the IR files and coverage summary; callers must validate before use."""


class BlockTokenizer(Protocol):
    """Minimal tokenizer surface needed by deterministic block rendering."""

    def text_token_ids(self, text: str) -> list[int]:
        """Encode content without adding model special tokens."""

        ...

    def detokenize_tokens(self, tokens: list[int]) -> str:
        """Decode token IDs into text without dropping special tokens."""

        ...


class DeterministicBlockRenderer:
    """Render hash blocks as exact-length deterministic text.

    Dataset-scoped IDs share text globally. Session-scoped IDs include the
    source session in their key, preventing accidental cross-session cache
    sharing while retaining exact prefix identity within one session.
    """

    version = "agentinfer-deterministic-block-text/v1"

    def __init__(
        self,
        tokenizer: BlockTokenizer,
        *,
        block_size: int,
        block_id_scope: Literal["session", "dataset"],
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.block_id_scope = block_id_scope
        self._cache: dict[tuple[str, int], str] = {}
        self._prefix_tokens: dict[tuple[str, tuple[str, ...]], tuple[int, ...]] = {}

    def _scope_key(self, session_id: str, block_id: str) -> str:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        return f"session:{session_id}:{block_id}" if self.block_id_scope == "session" else f"dataset:{block_id}"

    def render_block(self, session_id: str, block_id: str, token_count: int) -> str:
        """Render one block and verify detokenize/tokenize round-trip length."""

        if not block_id:
            raise ValueError("block_id must be non-empty")
        if not 0 < token_count <= self.block_size:
            raise ValueError("block token_count must be in (0, block_size]")
        scope_key = self._scope_key(session_id, block_id)
        cache_key = (scope_key, token_count)
        if cache_key in self._cache:
            return self._cache[cache_key]
        for variant in range(128):
            digest = hashlib.sha256(f"{self.version}\0{scope_key}\0{variant}".encode()).hexdigest()
            seed_tokens = self.tokenizer.text_token_ids(f" {digest}")
            if not seed_tokens:
                continue
            tokens = [seed_tokens[index % len(seed_tokens)] for index in range(token_count)]
            text = self.tokenizer.detokenize_tokens(tokens)
            if len(self.tokenizer.text_token_ids(text)) == token_count:
                self._cache[cache_key] = text
                return text
        raise ValueError(f"cannot render exact {token_count}-token block for {scope_key}")

    def render_request(self, session_id: str, hash_ids: list[str], input_length: int) -> str:
        """Render fixed full blocks plus one derived variable-length tail block."""

        if not hash_ids:
            raise ValueError("hash_ids must be non-empty")
        tail_length = input_length - self.block_size * (len(hash_ids) - 1)
        if not 0 < tail_length <= self.block_size:
            raise ValueError("input_length is inconsistent with hash_ids and block_size")
        parts = [
            self.render_block(
                session_id,
                block_id,
                self.block_size if index < len(hash_ids) - 1 else tail_length,
            )
            for index, block_id in enumerate(hash_ids)
        ]
        text = "".join(parts)
        token_ids = self.tokenizer.text_token_ids(text)
        if len(token_ids) != input_length:
            raise ValueError("rendered block boundaries do not round-trip to input_length")
        prefix_scope = session_id if self.block_id_scope == "session" else "dataset"
        for block_count in range(1, len(hash_ids)):
            boundary = block_count * self.block_size
            prefix_key = (prefix_scope, tuple(hash_ids[:block_count]))
            observed = tuple(token_ids[:boundary])
            expected = self._prefix_tokens.setdefault(prefix_key, observed)
            if observed != expected:
                raise ValueError("shared hash prefix does not preserve token identity at a block boundary")
        return text
