# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Read complete AgentX hash recipes and render deterministic Backend token IDs."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path


class HashSnapshotStore:
    """Look up a validated recipe by byte offset without indexing the entire file."""

    def __init__(self, requests_path: Path) -> None:
        self.requests_path = requests_path

    def read(self, offset: str, request_id: str) -> dict[str, object]:
        try:
            position = int(offset)
        except ValueError as exc:
            raise ValueError(f"invalid hash recipe offset for {request_id}") from exc
        if position < 0:
            raise ValueError(f"invalid hash recipe offset for {request_id}")
        with self.requests_path.open("rb") as handle:
            handle.seek(position)
            line = handle.readline()
        row = json.loads(line)
        if row.get("request_id") != request_id:
            raise ValueError(f"hash recipe offset does not match request {request_id}")
        return row


class TokenBlockRenderer:
    """Map local hash IDs to complete token blocks with bounded caching."""

    version = "agentinfer-agentx-token-blocks/v1"

    def __init__(self, palette: tuple[int, ...], *, cache_blocks: int = 8192) -> None:
        if len(set(palette)) < 2 or any(type(token) is not int or token < 0 for token in palette):
            raise ValueError("a token palette needs at least two distinct valid token IDs")
        if cache_blocks < 1:
            raise ValueError("cache_blocks must be positive")
        self.palette = tuple(dict.fromkeys(palette))
        self.cache_blocks = cache_blocks
        self.cache: OrderedDict[tuple[str, str, int, int], tuple[int, ...]] = OrderedDict()

    def render_block(self, runtime_session_id: str, source_model: str, hash_id: int, size: int) -> tuple[int, ...]:
        if not runtime_session_id or not source_model or type(hash_id) is not int or hash_id < 0 or size < 1:
            raise ValueError("invalid hash block identity or size")
        key = (runtime_session_id, source_model, hash_id, size)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        material = f"{self.version}\0{runtime_session_id}\0{source_model}\0{hash_id}".encode()
        values = bytearray()
        counter = 0
        while len(values) < size:
            values.extend(hashlib.sha256(material + counter.to_bytes(4, "big")).digest())
            counter += 1
        block = tuple(self.palette[value % len(self.palette)] for value in values[:size])
        self.cache[key] = block
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return block

    def render(self, recipe: dict[str, object], runtime_session_id: str) -> tuple[int, ...]:
        if recipe.get("hash_id_scope") != "local":
            raise ValueError("AgentX hash_id_scope must be local")
        size = recipe.get("block_size")
        input_tokens = recipe.get("input_tokens")
        hashes = recipe.get("hash_ids")
        source_model = recipe.get("source_model")
        if (
            type(size) is not int
            or size < 1
            or type(input_tokens) is not int
            or input_tokens < 1
            or not isinstance(hashes, list)
            or not hashes
            or len(hashes) != (input_tokens + size - 1) // size
            or any(type(value) is not int or value < 0 for value in hashes)
            or not isinstance(source_model, str)
            or not source_model
        ):
            raise ValueError("invalid AgentX hash snapshot recipe")
        tokens: list[int] = []
        for index, hash_id in enumerate(hashes):
            length = min(size, input_tokens - index * size)
            tokens.extend(self.render_block(runtime_session_id, source_model, hash_id, length))
        if len(tokens) != input_tokens:
            raise ValueError("rendered AgentX input token count differs from source")
        return tuple(tokens)
