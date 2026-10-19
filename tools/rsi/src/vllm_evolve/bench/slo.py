"""SLO definition, per-request attainment, and goodput.

This is the layer that decides what "good enough" means, kept separate from the
raw-measurement layer (:mod:`vllm_evolve.bench.metrics`). All pure functions.

**Goodput** is the headline serving metric: the rate of requests that *both*
completed successfully *and* met every specified SLO threshold. Reporting raw
throughput without an SLO is misleading because a policy can inflate throughput
by letting latency explode; goodput closes that loophole.

An SLO leaves a threshold unset (``None``) to mean "don't constrain this
dimension". A request *attains* the SLO iff it succeeded and every *specified*
threshold is satisfied (value <= threshold).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from vllm_evolve.bench.metrics import RequestRecord


@dataclass(frozen=True)
class SLO:
    """Service-level objective. Unset (``None``) thresholds are not enforced."""

    ttft_ms: float | None = None   # max acceptable time-to-first-token
    tpot_ms: float | None = None   # max acceptable time-per-output-token
    e2e_ms: float | None = None    # max acceptable end-to-end latency

    def is_empty(self) -> bool:
        """True if no threshold is set (attainment then == success)."""
        return self.ttft_ms is None and self.tpot_ms is None and self.e2e_ms is None


def attains(record: RequestRecord, slo: SLO) -> bool:
    """Whether a single request meets the SLO.

    A failed request never attains. For a successful request, every *specified*
    threshold must hold. TPOT is only checked when the request produced enough
    output tokens for TPOT to be defined (>= 2); a too-short request cannot
    violate a TPOT bound it never exercised.
    """
    if not record.success:
        return False
    if slo.ttft_ms is not None and record.ttft_ms > slo.ttft_ms:
        return False
    if slo.tpot_ms is not None and record.num_output_tokens >= 2 and record.tpot_ms > slo.tpot_ms:
        return False
    if slo.e2e_ms is not None and record.e2e_ms > slo.e2e_ms:
        return False
    return True


@dataclass
class SLOResult:
    """Outcome of applying an SLO to a run."""

    total: int = 0
    completed: int = 0
    attained: int = 0
    attainment_rate: float = 0.0     # attained / total
    goodput_req_s: float = 0.0       # attained requests / duration
    goodput_tok_s: float = 0.0       # output tokens of attained reqs / duration

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_slo(
    records: Sequence[RequestRecord],
    slo: SLO,
    duration_s: float,
) -> SLOResult:
    """Compute attainment and goodput for a run.

    ``attainment_rate`` is over *all* requests (a failed request counts against
    you). Goodput rates use the supplied wall-clock ``duration_s``; a
    non-positive duration yields zero goodput rather than dividing by zero.
    """
    total = len(records)
    completed = sum(1 for r in records if r.success)
    attained_records = [r for r in records if attains(r, slo)]
    attained = len(attained_records)
    attained_out_tokens = sum(r.num_output_tokens for r in attained_records)

    dur = duration_s if duration_s and duration_s > 0 else 0.0
    return SLOResult(
        total=total,
        completed=completed,
        attained=attained,
        attainment_rate=(attained / total) if total > 0 else 0.0,
        goodput_req_s=(attained / dur) if dur > 0 else 0.0,
        goodput_tok_s=(attained_out_tokens / dur) if dur > 0 else 0.0,
    )
