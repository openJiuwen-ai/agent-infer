"""Aggregate serving metrics from per-request records.

These are **pure functions** with no backend dependency: the real vLLM
backend (``bench/backend.py``) produces a ``list[RequestRecord]`` from an
actual run, and this module turns that list into a ``BenchMetrics`` object.
Because it is pure, it is fully unit-testable without a GPU.

Metric definitions follow standard LLM-serving conventions:

* **TTFT** (time-to-first-token, ms): wall time from request submission to the
  first output token. Lower is better.
* **TPOT** (time-per-output-token / inter-token latency, ms): mean gap between
  successive decode tokens. Lower is better. Undefined for requests with < 2
  output tokens (excluded from the TPOT distribution).
* **E2E** (end-to-end latency, ms): submission to final token.
* **request throughput** (req/s): completed requests / wall-clock duration.
* **output-token throughput** (tok/s): output tokens of completed requests /
  duration. This is the headline "throughput" number.
* **total-token throughput** (tok/s): (prompt + output) tokens / duration.

SLO-attainment goodput lives in :mod:`vllm_evolve.bench.slo`, not here, to keep
the raw-measurement layer separate from the policy/threshold layer.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

# Percentiles reported for every latency distribution.
_PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)


@dataclass
class RequestRecord:
    """One request's outcome from a real serving run.

    All latencies are milliseconds. ``arrival_time_s`` is the wall-clock second
    (relative to run start) at which the request was *submitted*; it is used by
    replay/throughput accounting, not by the latency percentiles.
    """

    request_id: str
    arrival_time_s: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_ms: float = 0.0
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0
    success: bool = True
    error: str | None = None


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, matching numpy's default method.

    ``sorted_values`` must already be sorted ascending. Returns 0.0 for an
    empty sequence and the single value for a length-1 sequence.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_values[0])
    rank = (n - 1) * (q / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_values[lo])
    frac = rank - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def _pct_map(values: list[float]) -> dict[str, float]:
    """Return {"p50": .., "p90": .., ...} for the given values."""
    s = sorted(values)
    return {f"p{p}": _percentile(s, p) for p in _PERCENTILES}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class BenchMetrics:
    """Aggregate metrics for one serving run over one workload."""

    num_requests: int = 0
    num_completed: int = 0
    num_failed: int = 0
    duration_s: float = 0.0

    request_throughput_req_s: float = 0.0
    output_throughput_tok_s: float = 0.0
    total_token_throughput_tok_s: float = 0.0

    ttft_ms: dict[str, float] = field(default_factory=dict)   # p50/p90/p95/p99
    tpot_ms: dict[str, float] = field(default_factory=dict)
    e2e_ms: dict[str, float] = field(default_factory=dict)

    mean_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    mean_e2e_ms: float = 0.0

    @classmethod
    def from_records(
        cls,
        records: Sequence[RequestRecord],
        duration_s: float,
    ) -> BenchMetrics:
        """Aggregate per-request records into one ``BenchMetrics``.

        Latency percentiles are computed over **completed** requests only.
        Throughput uses completed requests over the supplied wall-clock
        ``duration_s``. A non-positive duration yields zero throughput rather
        than dividing by zero.
        """
        completed = [r for r in records if r.success]
        failed = [r for r in records if not r.success]

        ttfts = [r.ttft_ms for r in completed]
        # TPOT is only defined when there were >= 2 output tokens.
        tpots = [r.tpot_ms for r in completed if r.num_output_tokens >= 2 and r.tpot_ms > 0]
        e2es = [r.e2e_ms for r in completed]

        out_tokens = sum(r.num_output_tokens for r in completed)
        all_tokens = sum(r.num_prompt_tokens + r.num_output_tokens for r in completed)

        dur = duration_s if duration_s and duration_s > 0 else 0.0
        if dur > 0:
            req_tput = len(completed) / dur
            out_tput = out_tokens / dur
            tot_tput = all_tokens / dur
        else:
            req_tput = out_tput = tot_tput = 0.0

        return cls(
            num_requests=len(records),
            num_completed=len(completed),
            num_failed=len(failed),
            duration_s=float(duration_s),
            request_throughput_req_s=req_tput,
            output_throughput_tok_s=out_tput,
            total_token_throughput_tok_s=tot_tput,
            ttft_ms=_pct_map(ttfts),
            tpot_ms=_pct_map(tpots),
            e2e_ms=_pct_map(e2es),
            mean_ttft_ms=_mean(ttfts),
            mean_tpot_ms=_mean(tpots),
            mean_e2e_ms=_mean(e2es),
        )

    def to_dict(self) -> dict:
        """JSON-serializable view (used by the eval_result archive)."""
        return asdict(self)
