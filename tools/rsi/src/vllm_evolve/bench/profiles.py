"""Benchmark profile + decision-policy loading.

A *profile* defines one benchmark regime (throughput / latency / replay): the
workload, the SLO, the primary metric, and the multi-seed schedule. Profiles are
YAML so they are human-editable; the SLO thresholds and the decision policy are
where the user expresses what "better" means for their deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vllm_evolve.bench.decision import DecisionPolicy
from vllm_evolve.bench.metrics import BenchMetrics
from vllm_evolve.bench.slo import SLO, SLOResult


@dataclass
class BenchProfile:
    name: str
    regime: str
    description: str = ""
    workload: dict = field(default_factory=dict)
    slo: SLO = field(default_factory=SLO)
    primary_metric: str = "goodput_req_s"
    primary_higher_is_better: bool = True
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
    seed_tiers: list[int] = field(default_factory=lambda: [3, 5, 10])
    cv_threshold: float | None = 0.10

    @classmethod
    def from_dict(cls, d: dict) -> BenchProfile:
        slo_d = d.get("slo") or {}
        return cls(
            name=d["name"],
            regime=d["regime"],
            description=d.get("description", ""),
            workload=dict(d.get("workload", {})),
            slo=SLO(
                ttft_ms=slo_d.get("ttft_ms"),
                tpot_ms=slo_d.get("tpot_ms"),
                e2e_ms=slo_d.get("e2e_ms"),
            ),
            primary_metric=d.get("primary_metric", "goodput_req_s"),
            primary_higher_is_better=bool(d.get("primary_higher_is_better", True)),
            seeds=list(d.get("seeds", [1, 2, 3, 5, 8, 13, 21, 34, 55, 89])),
            seed_tiers=list(d.get("seed_tiers", [3, 5, 10])),
            cv_threshold=d.get("cv_threshold", 0.10),
        )


def load_profile(path: str | Path) -> BenchProfile:
    with open(path, encoding="utf-8") as f:
        return BenchProfile.from_dict(yaml.safe_load(f))


def load_decision_policy(path: str | Path) -> DecisionPolicy:
    with open(path, encoding="utf-8") as f:
        return DecisionPolicy.from_dict(yaml.safe_load(f) or {})


def primary_metric_fn(name: str):
    """Resolve a profile's ``primary_metric`` name to an extractor function."""
    def extract(metrics: BenchMetrics, slo: SLOResult) -> float:
        if name == "goodput_req_s":
            return slo.goodput_req_s
        if name == "goodput_tok_s":
            return slo.goodput_tok_s
        if name == "output_throughput_tok_s":
            return metrics.output_throughput_tok_s
        if name == "request_throughput_req_s":
            return metrics.request_throughput_req_s
        if name in metrics.ttft_ms:        # e.g. "p99" against the ttft distribution
            return metrics.ttft_ms[name]
        raise KeyError(f"unknown primary_metric {name!r}")

    return extract
