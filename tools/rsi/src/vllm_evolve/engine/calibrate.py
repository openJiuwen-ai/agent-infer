"""M-B3 saturation calibration — find the "VRAM-full / max-throughput" operating point.

The sweep itself runs on real GPU (box-gated), but the *judgment* — is a point
saturated, has throughput reached a plateau, which point to lock — is pure and
unit-tested here. Saturation is NOT "mem_used/mem_total≥0.92" alone (Codex#11):
it requires GPU compute busy AND a throughput plateau AND evidence the client is
not the bottleneck. closed-loop (concurrency) vs open-loop (arrival rate) is
recorded so the regime is explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SweepPoint:
    """One measured operating point from a calibration sweep."""
    config: dict                       # the BenchConfig knobs for this point
    tok_s: float                       # measured output_throughput_tok_s
    sm_util_max: float = 0.0           # 0-100
    mem_used_mb: float = 0.0
    mem_total_mb: float = 1.0
    kv_util: float | None = None       # 0-1
    preempt: int | None = None
    client_saturated: bool | None = None   # True => client (not server) was the limit
    load_mode: str = "closed"          # closed | open
    eval_result: dict | None = None    # the full eval_result for this point (for freezing)

    def mem_frac(self) -> float:
        return self.mem_used_mb / self.mem_total_mb if self.mem_total_mb else 0.0


def is_saturated(p: SweepPoint, *, util_min: float = 90.0, mem_min: float = 0.92) -> bool:
    """A point is saturated only with sustained-high SM util AND near-full memory AND a
    PROVEN non-client bottleneck. ``client_saturated`` must be explicitly False (we
    checked the client wasn't the limit); None (unknown) does NOT pass — absence of
    evidence is not saturation (Codex MED#6). mem-full alone is never enough."""
    if p.client_saturated is not False:    # True (client-bound) or None (unproven) -> reject
        return False
    return p.sm_util_max >= util_min and p.mem_frac() >= mem_min


def throughput_plateaued(sweep: list[SweepPoint], *, rel_gain_eps: float = 0.03) -> bool:
    """True iff the last step of increasing load gained < rel_gain_eps throughput
    (further load does not raise tok/s -> we are on the plateau)."""
    if len(sweep) < 2:
        return False
    prev, last = sweep[-2].tok_s, sweep[-1].tok_s
    if prev <= 0:
        return False
    return (last - prev) / prev < rel_gain_eps


@dataclass
class CalibrationResult:
    saturated: bool
    locked_config: dict | None
    locked_tok_s: float | None
    reasons: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


def select_saturation_point(sweep: list[SweepPoint], *, util_min: float = 90.0,
                            mem_min: float = 0.92, rel_gain_eps: float = 0.03,
                            require_plateau: bool = True) -> CalibrationResult:
    """Lock the highest-throughput point that is saturated and on the plateau.

    Honest: if no point meets the criteria, returns saturated=False with reasons —
    the caller must NOT proceed as if a saturated regime were established.
    """
    reasons: list[str] = []
    sat = [p for p in sweep if is_saturated(p, util_min=util_min, mem_min=mem_min)]
    if not sat:
        reasons.append(f"no point reached util≥{util_min} + mem≥{mem_min} with a non-client "
                       "bottleneck")
        return CalibrationResult(False, None, None, reasons,
                                 {"n_points": len(sweep), "n_saturated": 0})
    plateau_ok = (not require_plateau) or throughput_plateaued(sweep, rel_gain_eps=rel_gain_eps)
    if require_plateau and not plateau_ok:
        reasons.append("throughput still rising — not on a plateau (more load may help)")
    best = max(sat, key=lambda p: p.tok_s)
    if not plateau_ok:
        return CalibrationResult(False, None, None, reasons,
                                 {"n_saturated": len(sat), "best_tok_s": best.tok_s})
    return CalibrationResult(
        True, dict(best.config), best.tok_s,
        ["saturated + plateau"],
        {"n_points": len(sweep), "n_saturated": len(sat), "load_mode": best.load_mode,
         "sm_util_max": best.sm_util_max, "mem_frac": round(best.mem_frac(), 3),
         "kv_util": best.kv_util, "preempt": best.preempt,
         "locked_eval_result": best.eval_result},   # for freezing the strong baseline
    )


def default_sweep_grid() -> list[dict]:
    """Deterministic closed-loop load sweep: rising concurrency at fixed high mem util.
    The real driver runs each point through the bench; this only defines the points."""
    return [{"concurrency": c, "gpu_memory_utilization": 0.9}
            for c in (32, 64, 128, 256, 512)]


def run_calibration_sweep(base_config: dict, eval_fn, grid: list[dict] | None = None,
                          **select_kw) -> CalibrationResult:
    """Drive a real saturation sweep. For each grid point, ``eval_fn(config)`` returns a
    dict of measured signals (tok_s, sm_util_max, mem_used_mb, mem_total_mb, kv_util,
    preempt, client_saturated, load_mode); the points feed ``select_saturation_point``.
    ``eval_fn`` is injected (real = the bench backend; tests pass a mock). Honest: if a
    point's eval raises (e.g. box unreachable), it is recorded as a failed point, never
    fabricated."""
    grid = grid if grid is not None else default_sweep_grid()
    points: list[SweepPoint] = []
    points_meta: list[dict] = []                # audit-grade per-point evidence (Codex R3)
    failures = 0
    for pt in grid:
        cfg = {**base_config, **pt}
        meta: dict = {"config": cfg}
        try:
            m = eval_fn(cfg)
        except Exception as exc:                # noqa: BLE001 - honest failed sweep point
            failures += 1
            meta["status"] = "failed"
            meta["error"] = str(exc)[-300:]
            points_meta.append(meta)
            continue
        meta["status"] = "ok"
        meta["signals"] = {k: v for k, v in m.items() if not str(k).startswith("_")}
        meta["rendered_serve_args"] = m.get("_rendered_serve_args")
        meta["remote_cmd"] = m.get("_remote_cmd")
        points_meta.append(meta)
        points.append(SweepPoint(
            config=cfg, tok_s=float(m.get("tok_s", 0.0)),
            sm_util_max=float(m.get("sm_util_max", 0.0)),
            mem_used_mb=float(m.get("mem_used_mb", 0.0)),
            mem_total_mb=float(m.get("mem_total_mb", 1.0)),
            kv_util=m.get("kv_util"), preempt=m.get("preempt"),
            client_saturated=m.get("client_saturated"),
            load_mode=m.get("load_mode", "closed"),
            eval_result=m.get("_eval_result")))
    result = select_saturation_point(points, **select_kw)
    result.evidence["points"] = points_meta
    result.evidence["sweep_failures"] = failures
    result.evidence["n_evaluated"] = len(points)
    return result
