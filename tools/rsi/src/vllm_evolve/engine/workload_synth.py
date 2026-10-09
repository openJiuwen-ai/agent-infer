"""R3 — synthesize Frontier ``trace_replay`` CSVs (P1's workload primitive).

The experiment workload primitive is a TRACE: a CSV with Frontier's native 5-column contract
``arrived_at,num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids`` (confirmed against the
real ``trace_replay_request_generator``). "Tenant" maps natively to ``session_id`` + a shared
``block_hash_ids`` prefix per tenant (so bursty multi-tenant load produces real prefix-cache reuse,
not an approximation).

Named shapes (``bursty_multi_tenant``, ``steady``) are TEMPLATE GENERATORS — convenience functions,
NOT the accepted universe. An agent may submit any trace whose rows satisfy ``validate_trace``'s
RANGE checks (row count / token range / duration cap). The guardrail is a budget+range, never an
enum of allowed shapes (C4: do not recreate the closed world under a new field name).

Deterministic: no RNG — same params yield identical rows (reproducible experiments). Pure data
generation; imports nothing from ``core``/``bench``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

TRACE_COLUMNS = (
    "arrived_at", "num_prefill_tokens", "num_decode_tokens", "session_id", "block_hash_ids",
)


@dataclass
class TraceRanges:
    """Range guardrail for an arbitrary trace (NOT an enum of shapes). A trace is admissible iff
    every row's tokens are in range, the row count is within ``max_rows``, and the last arrival is
    within ``max_duration_s``."""

    max_rows: int = 5000
    prefill_range: tuple = (1, 8192)
    decode_range: tuple = (1, 4096)
    max_duration_s: float = 600.0


def bursty_multi_tenant(
    *,
    n_tenants: int = 2,
    bursts_per_tenant: int = 3,
    burst_size: int = 4,
    burst_period_s: float = 1.0,
    intra_burst_gap_s: float = 0.01,
    tenant_stagger_s: float = 0.05,
    prefill_tokens: int = 512,
    decode_tokens: int = 64,
    prefix_blocks_per_tenant: int = 2,
) -> list:
    """A deterministic bursty multi-tenant trace: each tenant emits ``bursts_per_tenant`` bursts of
    ``burst_size`` requests; tenants are staggered so bursts overlap (contention). Every request
    from a tenant shares that tenant's ``block_hash_ids`` prefix blocks -> repeated prompt blocks ->
    prefix-cache reuse (and, under bursts, thrash). Rows returned sorted by ``arrived_at``."""
    rows: list[dict] = []
    for burst in range(bursts_per_tenant):
        burst_start = burst * burst_period_s
        for tenant in range(n_tenants):
            blocks = "|".join(str(tenant * 1000 + k) for k in range(prefix_blocks_per_tenant))
            base = burst_start + tenant * tenant_stagger_s
            for i in range(burst_size):
                rows.append({
                    "arrived_at": round(base + i * intra_burst_gap_s, 6),
                    "num_prefill_tokens": int(prefill_tokens),
                    "num_decode_tokens": int(decode_tokens),
                    "session_id": int(tenant),
                    "block_hash_ids": blocks,
                })
    rows.sort(key=lambda r: (r["arrived_at"], r["session_id"]))
    return rows


def heterogeneous_bursty_multi_tenant(
    *,
    n_tenants: int = 4,
    bursts_per_tenant: int = 3,
    burst_size: int = 5,
    burst_period_s: float = 0.75,
    intra_burst_gap_s: float = 0.015,
    tenant_stagger_s: float = 0.05,
    length_profiles: tuple[tuple[int, int], ...] = (
        (256, 2048),
        (768, 64),
        (1024, 96),
        (1536, 2048),
    ),
    prefix_blocks_per_tenant: int = 2,
) -> list:
    """Bursty trace with deterministic, anti-correlated prompt/output lengths.

    Both prompt-length extremes include a long-decode profile. This avoids making either SJF or
    LJF the answer by construction: each can admit a request that then occupies a sequence slot
    for a long decode, while completion-aware policies can release slots sooner. This is a
    separately labeled synthetic stress workload, never represented as BurstGPT data.
    """
    if not length_profiles:
        raise ValueError("length_profiles must not be empty")
    rows = bursty_multi_tenant(
        n_tenants=n_tenants,
        bursts_per_tenant=bursts_per_tenant,
        burst_size=burst_size,
        burst_period_s=burst_period_s,
        intra_burst_gap_s=intra_burst_gap_s,
        tenant_stagger_s=tenant_stagger_s,
        prefill_tokens=1,
        decode_tokens=1,
        prefix_blocks_per_tenant=prefix_blocks_per_tenant,
    )
    for index, row in enumerate(rows):
        prompt, output = length_profiles[index % len(length_profiles)]
        row["num_prefill_tokens"] = int(prompt)
        row["num_decode_tokens"] = int(output)
    return rows


def steady(
    *, n_requests: int = 16, gap_s: float = 0.1,
    prefill_tokens: int = 512, decode_tokens: int = 64, session_id: int = 0,
) -> list:
    """A single-tenant steady-arrival trace (template; baseline workload)."""
    return [{
        "arrived_at": round(i * gap_s, 6),
        "num_prefill_tokens": int(prefill_tokens),
        "num_decode_tokens": int(decode_tokens),
        "session_id": int(session_id),
        "block_hash_ids": "0",
    } for i in range(n_requests)]


def _to_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def validate_trace(rows: list, ranges: TraceRanges | None = None) -> list:
    """Return human-readable RANGE/TYPE violations (empty == admissible). Range guardrail only —
    never an enum of allowed shapes, and **never raises** on malformed open-world input: each field
    is validated independently, a non-numeric value becomes an explicit error, and scanning
    continues so the caller gets the complete violation list."""
    rng = ranges or TraceRanges()
    errors: list[str] = []
    if not rows:
        return ["empty trace"]
    if len(rows) > rng.max_rows:
        errors.append(f"row count {len(rows)} > max_rows {rng.max_rows}")
    arrivals: list[float] = []
    for i, r in enumerate(rows):
        missing = set(TRACE_COLUMNS) - set(r)
        if missing:
            errors.append(f"row {i} missing columns {sorted(missing)}")
            continue
        pt = _to_int(r["num_prefill_tokens"])
        if pt is None:
            errors.append(f"row {i} num_prefill_tokens not an integer: {r['num_prefill_tokens']!r}")
        elif not rng.prefill_range[0] <= pt <= rng.prefill_range[1]:
            errors.append(f"row {i} prefill {pt} out of range {rng.prefill_range}")
        dt = _to_int(r["num_decode_tokens"])
        if dt is None:
            errors.append(f"row {i} num_decode_tokens not an integer: {r['num_decode_tokens']!r}")
        elif not rng.decode_range[0] <= dt <= rng.decode_range[1]:
            errors.append(f"row {i} decode {dt} out of range {rng.decode_range}")
        at = _to_float(r["arrived_at"])
        if at is None:
            errors.append(f"row {i} arrived_at not a number: {r['arrived_at']!r}")
        else:
            arrivals.append(at)
    if not arrivals:
        errors.append("no valid numeric arrived_at in trace")
    elif max(arrivals) > rng.max_duration_s:
        errors.append(f"trace duration {max(arrivals)}s > max_duration_s {rng.max_duration_s}")
    return errors


def write_trace_csv(rows: list, path: str | Path) -> Path:
    """Write rows to a Frontier ``trace_replay`` CSV (the 5-column contract). Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TRACE_COLUMNS))
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r[c] for c in TRACE_COLUMNS})
    return p


@dataclass
class WorkloadSpec:
    """Declarative workload for an ExperimentSpec (R1 consumes this): a named template + params, or
    an explicit inline trace. The template is sugar; the admitted thing is the resulting trace +
    range validation."""

    template: str | None = None                 # "bursty_multi_tenant" | "steady" | None
    params: dict = field(default_factory=dict)
    inline_rows: list = field(default_factory=list)

    def build(self) -> list:
        """Materialize the trace rows (template call or inline)."""
        if self.inline_rows:
            return list(self.inline_rows)
        templates = {
            "bursty_multi_tenant": bursty_multi_tenant,
            "heterogeneous_bursty_multi_tenant": heterogeneous_bursty_multi_tenant,
            "steady": steady,
        }
        if self.template not in templates:
            raise ValueError(
                f"unknown workload template {self.template!r}; pass inline_rows for a custom trace")
        return templates[self.template](**(self.params or {}))
