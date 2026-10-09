"""ShareGPT trace adapter (ShareGPT_V3_unfiltered_cleaned_split.json).

Format: a single UTF-8 JSON **array** of conversation objects (NOT JSONL). Each
record looks like ``{"id": ..., "conversations": [{"from": "human", "value":
...}, {"from": "gpt", "value": ...}, ...]}``. Following vLLM's ShareGPTDataset:
records without a ``conversations`` key or with < 2 turns are dropped; turn[0]
is the prompt and turn[1] is the completion (the role label is ignored).

ShareGPT stores **no token counts and no arrival timestamps**. Token counts are
derived with a tokenizer; if none is supplied we fall back to a ~4-chars/token
heuristic (and log a warning). Arrival times are synthesized as a 1-req/s
placeholder here — the replay load profile re-schedules arrivals (e.g. Poisson
at a target request rate), so the placeholder value is not load-bearing.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from vllm_evolve.bench.datasets.base import (
    TraceDataset,
    TraceRequest,
    char_token_estimate,
)

_log = logging.getLogger(__name__)


def load(
    path: str | Path,
    *,
    count_tokens: Callable[[str], int] | None = None,
    output_len: int | None = None,
    max_requests: int | None = None,
    name: str = "sharegpt",
) -> TraceDataset:
    """Load ShareGPT conversations into a TraceDataset.

    ``count_tokens`` maps text -> token count (wrap a real tokenizer here). When
    ``None``, the char heuristic is used. ``output_len``, if given, overrides the
    derived completion length for every request (matches vLLM's
    ``--sharegpt-output-len``).
    """
    path = Path(path)
    if count_tokens is None:
        _log.warning(
            "sharegpt: no tokenizer supplied; using ~4-chars/token heuristic "
            "(token counts are approximate)"
        )
        count_tokens = char_token_estimate

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    requests: list[TraceRequest] = []
    errors = 0
    for idx, entry in enumerate(data):
        if max_requests is not None and len(requests) >= max_requests:
            break
        try:
            convs = entry["conversations"]
            if not isinstance(convs, list) or len(convs) < 2:
                continue
            prompt_text = convs[0]["value"]
            completion_text = convs[1]["value"]
        except (KeyError, TypeError, IndexError):
            errors += 1
            continue
        prompt_tokens = count_tokens(prompt_text)
        out_tokens = output_len if output_len is not None else count_tokens(completion_text)
        requests.append(
            TraceRequest(
                arrival_s=float(len(requests)),  # placeholder; replay re-schedules
                prompt_tokens=prompt_tokens,
                output_tokens=out_tokens,
                request_id=str(entry.get("id") or f"sharegpt_{idx:07d}"),
            )
        )
    return TraceDataset(name=name, requests=requests, parse_errors=errors)
