# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Reuse native vLLM page hashes for exactly matching Program input prefixes."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from agentinfer.agentcache.mori.programs import PROGRAM_ARG

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.request import Request


class MORIProgramBlockHasher:
    def __init__(self, native: Callable[[Request], list[BlockHash]], page_size: int, capacity_tokens: int) -> None:
        self._native = native
        self._page_size = page_size
        self._capacity_tokens = capacity_tokens
        self._cached_tokens = 0
        self._recent: OrderedDict[str, tuple[tuple[int, ...], tuple[BlockHash, ...]]] = OrderedDict()

    def __call__(self, request: Request) -> list[BlockHash]:
        # Decode updates already resume from the request's own hash chain.
        if request.block_hashes or request.num_tokens != request.num_prompt_tokens:
            return self._native(request)
        params = request.sampling_params
        program = (params.extra_args or {}).get(PROGRAM_ARG) if params is not None else None
        ids = request.prompt_token_ids
        if (
            not isinstance(program, str)
            or not program
            or ids is None
            or not self._page_size <= len(ids) <= self._capacity_tokens
            or request.cache_salt is not None
            or request.lora_request is not None
            or request.mm_features
            or request.prompt_embeds is not None
        ):
            return self._native(request)

        tokens = tuple(ids)
        hashes = []
        if previous := self._recent.get(program):
            pages = self._common_pages(previous[0], tokens)
            hashes = list(previous[1][:pages])
        if hashes:
            initial = request.block_hashes
            request.block_hashes = hashes
            try:
                hashes.extend(self._native(request))
            finally:
                request.block_hashes = initial
        else:
            hashes = self._native(request)

        # Retain only enough previous input to cover the two configured tiers.
        # The native allocator continues to decide which hashed pages are live.
        if previous:
            self._cached_tokens -= len(previous[0])
            self._recent.pop(program)
        while self._cached_tokens + len(tokens) > self._capacity_tokens:
            _, evicted = self._recent.popitem(last=False)
            self._cached_tokens -= len(evicted[0])
        self._recent[program] = (tokens, tuple(hashes))
        self._cached_tokens += len(tokens)
        return hashes

    def _common_pages(self, previous: tuple[int, ...], tokens: tuple[int, ...]) -> int:
        page = self._page_size
        end = min(len(previous), len(tokens)) // page * page
        # Compare large slices first so Python enters the page loop only around
        # the first divergence. Every reused page still matches exact token IDs.
        span = page * 256
        for start in range(0, end, span):
            stop = min(start + span, end)
            if previous[start:stop] == tokens[start:stop]:
                continue
            for boundary in range(start, stop, page):
                if previous[boundary : boundary + page] != tokens[boundary : boundary + page]:
                    return boundary // page
        return end // page
