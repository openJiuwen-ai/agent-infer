"""Azure LLM inference trace adapter (Azure/AzurePublicDataset).

Format: UTF-8 CSV, header ``TIMESTAMP,ContextTokens,GeneratedTokens`` (both the
2023 Splitwise and 2024 DynamoLLM releases share this 3-column schema).

``TIMESTAMP`` is an **absolute datetime string** ``YYYY-MM-DD HH:MM:SS.fffffff``
with **7 fractional-second digits** (.NET ticks). Python's ``%f`` accepts only 6
digits, so the 7th is truncated. The relative arrival offset (seconds) is the
difference from the earliest timestamp. ``ContextTokens`` = prompt length,
``GeneratedTokens`` = output length. No prefix/KV-reuse field.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from vllm_evolve.bench.datasets.base import TraceDataset, TraceRequest, resolve_column

_TS = {"timestamp", "time"}
_PROMPT = {"contexttokens", "context_tokens", "context", "prompt_tokens", "input_tokens"}
_OUTPUT = {"generatedtokens", "generated_tokens", "output_tokens", "completion_tokens"}


def _parse_ts(s: str) -> datetime:
    """Parse the Azure datetime, truncating 7-digit fractional seconds to 6."""
    s = s.strip()
    if "." in s:
        main, frac = s.split(".", 1)
        frac = (frac + "000000")[:6]  # truncate/pad to microseconds
        return datetime.strptime(f"{main}.{frac}", "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def load(
    path: str | Path,
    *,
    max_requests: int | None = None,
    name: str = "azure",
) -> TraceDataset:
    path = Path(path)
    raw: list[tuple[datetime, int, int, int]] = []
    errors = 0
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"empty Azure file: {path}")
        ts_i = resolve_column(header, _TS)
        p_i = resolve_column(header, _PROMPT)
        o_i = resolve_column(header, _OUTPUT)
        for idx, row in enumerate(reader):
            if max_requests is not None and len(raw) >= max_requests:
                break
            try:
                dt = _parse_ts(row[ts_i])
                prompt = int(row[p_i])
                output = int(row[o_i])
            except (IndexError, ValueError):
                errors += 1
                continue
            raw.append((dt, prompt, output, idx))

    requests: list[TraceRequest] = []
    if raw:
        t0 = min(r[0] for r in raw)
        for dt, prompt, output, idx in raw:
            requests.append(
                TraceRequest(
                    arrival_s=(dt - t0).total_seconds(),
                    prompt_tokens=prompt,
                    output_tokens=output,
                    request_id=f"azure_{idx:07d}",
                )
            )
    return TraceDataset(name=name, requests=requests, parse_errors=errors).sorted_by_arrival()
