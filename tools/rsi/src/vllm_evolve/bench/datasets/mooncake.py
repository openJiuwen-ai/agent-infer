"""Mooncake trace adapter (kvcache-ai/Mooncake).

Format: UTF-8 JSONL (one JSON object per line, no header, no enclosing array).
Fields: ``timestamp, input_length, output_length, hash_ids``. All four trace
variants (mooncake/conversation/synthetic/toolagent) share this schema.

``timestamp`` is a **relative offset in milliseconds** (÷1000 → seconds), not an
epoch. ``input_length`` = prompt tokens, ``output_length`` = output tokens.
``hash_ids`` is the prefix-cache chain: an ordered list of integer block-hash
IDs (block size 512 tokens); two requests sharing a leading run of ids share
that KV-cache prefix. Unknown extra keys (AIPerf extensions tools/messages/etc.)
are ignored. Malformed lines are skipped and counted.
"""
from __future__ import annotations

import json
from pathlib import Path

from vllm_evolve.bench.datasets.base import TraceDataset, TraceRequest


def load(
    path: str | Path,
    *,
    max_requests: int | None = None,
    name: str = "mooncake",
) -> TraceDataset:
    path = Path(path)
    requests: list[TraceRequest] = []
    errors = 0
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_requests is not None and len(requests) >= max_requests:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                arrival_ms = float(obj["timestamp"])
                prompt = int(obj["input_length"])
                output = int(obj["output_length"])
                hashes = tuple(int(h) for h in obj.get("hash_ids", []) or [])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                errors += 1
                continue
            requests.append(
                TraceRequest(
                    arrival_s=arrival_ms / 1000.0,
                    prompt_tokens=prompt,
                    output_tokens=output,
                    request_id=f"mooncake_{idx:07d}",
                    prefix_hashes=hashes,
                )
            )
    return TraceDataset(name=name, requests=requests, parse_errors=errors).sorted_by_arrival()
