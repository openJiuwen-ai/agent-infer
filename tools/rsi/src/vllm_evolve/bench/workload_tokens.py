"""Deterministic exact-length prompt-token construction for real benchmarks."""
from __future__ import annotations

import hashlib

PREFIX_BLOCK_TOKENS = 16


def _token_for(value: str, *, offset: int) -> int:
    digest = hashlib.sha256(value.encode()).digest()
    return offset + int.from_bytes(digest[:2], "big") % 700


def prompt_token_ids(row: dict, request_index: int) -> list[int]:
    """Build an exact-length prompt with only explicitly labeled shared prefixes."""
    length = int(row["num_prefill_tokens"])
    if length < 1:
        raise ValueError("real-vLLM prompt length must be positive")
    block_ids = [
        item
        for item in str(row.get("block_hash_ids") or "").split("|")
        if item
    ]
    tokens: list[int] = []
    for block_id in block_ids:
        token = _token_for(f"prefix:{block_id}", offset=100)
        tokens.extend([token] * min(PREFIX_BLOCK_TOKENS, length - len(tokens)))
        if len(tokens) >= length:
            break
    unique = _token_for(f"request:{request_index}", offset=1000)
    tokens.extend([unique] * (length - len(tokens)))
    return tokens

