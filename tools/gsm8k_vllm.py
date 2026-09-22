#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Run a reproducible GSM8K accuracy check through a vLLM OpenAI API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import httpx

DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
)
_ANSWER_RE = re.compile(r"####\s*([-+]?(?:\d[\d,]*\.?\d*|\.\d+))")


def _normalise_number(value: str) -> str | None:
    """Return a canonical decimal string for GSM8K's numeric answers."""

    try:
        number = Decimal(value.replace(",", "").strip())
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalised = format(number, "f").rstrip("0").rstrip(".")
    return normalised or "0"


def extract_answer(text: str) -> str | None:
    """Extract the final answer following GSM8K's ``####`` marker."""

    matches = _ANSWER_RE.findall(text)
    return _normalise_number(matches[-1]) if matches else None


def expected_answer(answer: str) -> str | None:
    """Extract the reference answer from one GSM8K explanation."""

    return extract_answer(answer)


def load_rows(url: str, cache_path: Path | None) -> tuple[list[dict[str, str]], str]:
    """Load the official test split and return rows plus its content SHA256."""

    if cache_path is not None and cache_path.exists():
        content = cache_path.read_bytes()
    else:
        with urlopen(url, timeout=60) as response:  # noqa: S310 - explicit benchmark URL
            content = response.read()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(content)
    rows = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("question"), str)
            or not isinstance(row.get("answer"), str)
        ):
            raise ValueError(f"invalid GSM8K row at line {line_number}")
        rows.append({"question": row["question"], "answer": row["answer"]})
    return rows, hashlib.sha256(content).hexdigest()


@dataclass
class Result:
    index: int
    question: str
    reference: str | None
    output: str
    prediction: str | None
    correct: bool
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


async def _request(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    row: dict[str, str],
    index: int,
    max_tokens: int,
    seed: int,
    retries: int,
) -> Result:
    prompt = (
        "Solve the following grade-school math problem. Show concise reasoning and finish with "
        "the exact numeric answer in the form #### <number>.\n\n" + row["question"]
    )
    reference = expected_answer(row["answer"])
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "seed": seed + index,
        "stream": False,
    }
    started = time.perf_counter()
    error: str | None = None
    output = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    for attempt in range(retries + 1):
        try:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            message = choice.get("message", {})
            output = "\n".join(
                str(value) for value in (message.get("reasoning_content"), message.get("content")) if value
            )
            usage = body.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            error = None
            break
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))
    prediction = extract_answer(output)
    return Result(
        index=index,
        question=row["question"],
        reference=reference,
        output=output,
        prediction=prediction,
        correct=error is None and prediction is not None and prediction == reference,
        latency_seconds=time.perf_counter() - started,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error=error,
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_path = Path(args.dataset_cache) if args.dataset_cache else None
    rows, dataset_sha256 = load_rows(args.dataset_url, cache_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    endpoint = args.endpoint.rstrip("/") + "/v1/chat/completions"
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:

        async def one(index: int, row: dict[str, str]) -> Result:
            async with semaphore:
                return await _request(
                    client,
                    endpoint,
                    args.model,
                    row,
                    index,
                    args.max_tokens,
                    args.seed,
                    args.retries,
                )

        results = await asyncio.gather(*(one(index, row) for index, row in enumerate(rows)))

    latencies = [result.latency_seconds for result in results]
    errors = [result for result in results if result.error is not None]
    parsed = [result for result in results if result.prediction is not None]
    summary = {
        "dataset_url": args.dataset_url,
        "dataset_sha256": dataset_sha256,
        "rows": len(results),
        "correct": sum(result.correct for result in results),
        "accuracy": (sum(result.correct for result in results) / len(results)) if results else 0.0,
        "parsed_predictions": len(parsed),
        "errors": len(errors),
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": statistics.median(latencies) if latencies else None,
            "p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else None,
        },
        "prompt_tokens": sum(result.prompt_tokens or 0 for result in results),
        "completion_tokens": sum(result.completion_tokens or 0 for result in results),
        "request": {
            "endpoint": endpoint,
            "model": args.model,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "top_p": 1,
            "top_k": 0,
            "seed": args.seed,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
        for result in results:
            handle.write(json.dumps({"type": "item", **result.as_dict()}, ensure_ascii=False) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--dataset-cache", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


def main() -> int:
    summary = asyncio.run(run(parse_args()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
