"""M-B4 (parser half) — parse a CUDA profile into a structured ``KernelBreakdown``.

The COLLECTOR (running ``nsys``/``ncu``/``torch.profiler`` against a real serve
window) is box-gated and lives in ``cuda_profile.py``; this module is the pure,
unit-tested PARSER. The torch-profiler **chrome trace** (JSON) is the grounded
format here — kernels are classified by name into attention / GEMM(linear) / KV /
comm(NCCL) / other, and time fractions feed the KernelToLeverMap (M-D1). nsys
sqlite / ncu csv parsing are version-sensitive and recorded as box-gated stubs
that degrade honestly (return ``None`` signals) rather than guess.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# kernel-name -> category. First match wins; case-insensitive.
_CATEGORY_PATTERNS: list[tuple[str, str]] = [
    (r"flash|attention|mha|sdpa|fmha|paged_attention", "attention"),
    (r"gemm|matmul|cutlass|linear|cublas|wgrad|sgemm|hgemm", "gemm"),
    (r"reshape_and_cache|kv_cache|copy_blocks|paged|cache_kernel", "kv"),
    (r"nccl|all_?reduce|all_?gather|reduce_scatter|broadcast", "comm"),
]


def classify_kernel(name: str) -> str:
    n = (name or "").lower()
    for pat, cat in _CATEGORY_PATTERNS:
        if re.search(pat, n):
            return cat
    return "other"


@dataclass
class KernelBreakdown:
    """Structured kernel-level summary; feeds match_levers via :meth:`to_signals`."""
    attention_time_frac: float | None = None
    gemm_time_frac: float | None = None
    kv_time_frac: float | None = None
    comm_time_frac: float | None = None
    other_time_frac: float | None = None
    total_kernel_ms: float | None = None
    # roofline-ish signals (from ncu when available; None from a plain chrome trace)
    tc_util: float | None = None
    dram_bw_util: float | None = None
    sm_occupancy: float | None = None
    launch_gap_frac: float | None = None
    source: str = ""

    def to_signals(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and k != "source"}

    def to_dict(self) -> dict:
        return asdict(self)


def parse_torch_trace(trace: dict) -> KernelBreakdown:
    """Parse a torch.profiler chrome trace into a KernelBreakdown.

    Uses only GPU kernel events (``cat`` containing 'kernel'/'gpu', with a duration).
    Time fractions are over total GPU-kernel time. CPU/launch events are ignored
    for the fractions (launch_gap needs timeline reconstruction -> left None here).
    """
    events = (trace or {}).get("traceEvents") or []
    by_cat: dict[str, float] = {"attention": 0.0, "gemm": 0.0, "kv": 0.0,
                                "comm": 0.0, "other": 0.0}
    total = 0.0
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = str(e.get("cat", "")).lower()
        dur = e.get("dur")
        if "kernel" not in cat and "gpu" not in cat:
            continue
        if not isinstance(dur, (int, float)):
            continue
        by_cat[classify_kernel(e.get("name", ""))] += float(dur)
        total += float(dur)
    if total <= 0:
        return KernelBreakdown(source="torch_trace")   # no kernels -> all None signals
    return KernelBreakdown(
        attention_time_frac=round(by_cat["attention"] / total, 4),
        gemm_time_frac=round(by_cat["gemm"] / total, 4),
        kv_time_frac=round(by_cat["kv"] / total, 4),
        comm_time_frac=round(by_cat["comm"] / total, 4),
        other_time_frac=round(by_cat["other"] / total, 4),
        total_kernel_ms=round(total / 1000.0, 4),       # chrome dur is microseconds
        source="torch_trace",
    )


def parse_ncu_csv(_csv_text: str) -> KernelBreakdown:  # pragma: no cover - box-gated
    """Parse ``ncu --csv`` output (occupancy / DRAM bw / tensor-core util). Box-gated;
    version-sensitive — returns an empty breakdown until validated on real ncu output."""
    return KernelBreakdown(source="ncu_csv_unparsed")
