"""Deterministic quality measurements and the frozen no-regression gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class QualityMeasurement:
    """Quality of one config measured on the same fixed calibration set."""

    perplexity: float | None = None
    task_em: float | None = None
    output_agreement: float | None = None


@dataclass
class QualityThresholds:
    """Pre-registered thresholds; callers may disable a gate only with ``None``."""

    max_perplexity_increase_pct: float | None = 1.0
    max_em_drop_pct: float | None = 1.0
    min_output_agreement: float | None = 0.99


@dataclass
class QualityVerdict:
    ok: bool
    reasons: list
    measured: dict

    def to_dict(self) -> dict:
        return asdict(self)


def quality_ok(
    baseline: QualityMeasurement,
    candidate: QualityMeasurement,
    thresholds: QualityThresholds | None = None,
) -> QualityVerdict:
    """Fail closed when an enabled quality signal is missing or regresses."""
    selected = thresholds or QualityThresholds()
    reasons: list[str] = []

    if selected.max_perplexity_increase_pct is not None:
        if baseline.perplexity is None or candidate.perplexity is None:
            reasons.append("perplexity not measured — cannot certify no regression")
        else:
            denom = baseline.perplexity if baseline.perplexity else 1e-9
            rise = 100.0 * (candidate.perplexity - baseline.perplexity) / denom
            if rise > selected.max_perplexity_increase_pct:
                reasons.append(
                    f"perplexity +{rise:.2f}% > {selected.max_perplexity_increase_pct}%"
                )

    if selected.max_em_drop_pct is not None:
        if baseline.task_em is None or candidate.task_em is None:
            reasons.append("task EM not measured on both — cannot certify (gate enabled)")
        else:
            denom = baseline.task_em if baseline.task_em else 1e-9
            drop = 100.0 * (baseline.task_em - candidate.task_em) / denom
            if drop > selected.max_em_drop_pct:
                reasons.append(f"task EM -{drop:.2f}% > {selected.max_em_drop_pct}%")

    if selected.min_output_agreement is not None:
        if candidate.output_agreement is None:
            reasons.append("output agreement not measured — cannot certify (gate enabled)")
        elif candidate.output_agreement < selected.min_output_agreement:
            reasons.append(
                f"output agreement {candidate.output_agreement:.3f} "
                f"< {selected.min_output_agreement}"
            )

    return QualityVerdict(
        ok=not reasons,
        reasons=reasons,
        measured={"baseline": asdict(baseline), "candidate": asdict(candidate)},
    )
