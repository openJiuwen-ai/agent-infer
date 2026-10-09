"""autopt layer-to-layer data contracts.

Plain dataclasses with ``to_dict`` / ``from_dict`` (JSON round-trippable) so each
``ar`` verb emits/consumes a stable structured object. Nothing here touches a GPU
or fabricates a value — these are pure carriers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# Bottleneck taxonomy (L1 diagnose output) — see design §3 L1 rule table.
COMPUTE = "compute"                 # GPU FLOPS bound (SM util high, KV not full)
KV_CAPACITY = "kv_capacity"         # KV cache full -> preemption / waiting grows
MEM_BANDWIDTH = "mem_bandwidth"     # weight/KV byte movement bound (low SM, high BW)
PREFILL = "prefill"                 # prefill dominates (TTFT high, long prompts)
SCHEDULING_QUEUE = "scheduling_queue"  # capacity exists but admission/order limits
UNDER_SATURATED = "under_saturated"    # GPU idle / not enough load
COMM = "comm"                       # TP/PP communication (NCCL) bound
UNKNOWN = "unknown"                 # evidence insufficient to localize

BOTTLENECKS = (
    COMPUTE, KV_CAPACITY, MEM_BANDWIDTH, PREFILL, SCHEDULING_QUEUE,
    UNDER_SATURATED, COMM, UNKNOWN,
)


@dataclass
class Spec:
    """L0: a measurable optimization goal parsed from a natural-language intent."""
    metric: str                      # e.g. "goodput_req_s" / "ttft_p99_ms" / "tok_s"
    target: float | None = None      # numeric target, or None for pure max/min
    direction: str = "max"           # "max" | "min" | "threshold"
    constraints: dict = field(default_factory=dict)   # {slo, cost, mem, ...}
    workload: dict = field(default_factory=dict)       # {kind, sizes, qps/arrival}
    hardware: dict = field(default_factory=dict)       # {gpus, model, tp}
    raw_intent: str = ""
    # NOTE: the accept-gate threshold deliberately does NOT live on Spec — it is a config/CLI
    # decision (run_autopt's accept_threshold_pct / the deterministic default), so a (possibly
    # sub-agent-produced) goal can never lower the adoption bar (Codex/skeptic M6).

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Spec:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Profile:
    """L1: a measured snapshot of one deployment/config under a workload.

    All numeric fields are Optional — a missing signal stays ``None`` (the
    diagnoser then degrades the rules that need it to ``unknown`` rather than
    guessing). Nothing is fabricated.
    """
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)   # ttft_p50_ms, ttft_p99_ms, tpot_ms,
                                                  # goodput_req_s, tok_s, num_completed...
    vllm: dict = field(default_factory=dict)      # running, waiting, kv_util(0-1),
                                                  # preempt, prefix_hit, prefill_time_frac
    gpu: dict = field(default_factory=dict)       # sm_util(0-100), mem_used_mb,
                                                  # mem_total_mb, mem_bw_util(0-100), nccl_frac
    per_seed: dict = field(default_factory=dict)  # metric -> [per-seed values] (for L4 bootstrap)
    saturated: bool | None = None
    outcome_class: str = ""
    source: str = ""                              # eval_result source: 'real_vllm' | 'local_smoke'
    marker_verified: bool = False                 # plugin actually invoked (anti-fabrication)
    bench_config: dict | None = None              # BenchConfig provenance (for same_caliber A/B)
    eval_result: dict | None = None               # the raw eval_result (for calibrate freezing)
    remote_cmd: str = ""                          # the ssh command line, for per-point evidence
    evidence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Profile:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TargetPlan:
    """L2: ranked candidate interventions for a diagnosed bottleneck."""
    bottleneck: str
    status: str                      # carried from the Diagnosis (confirmed/suspected/unknown)
    candidates: list = field(default_factory=list)   # [{target, type, leverage, cost, ...}]
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TargetPlan:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GainVerdict:
    """L4: is a candidate a real, robust win? (bootstrap + holdout + re-profile)."""
    verdict: str                     # better|worse|inconclusive|high_variance_inconclusive|
                                     # rejected_unverified
    improvement_pct: float | None = None
    ci_low_pct: float | None = None
    ci_high_pct: float | None = None
    candidate_cv: float | None = None
    holdout_ok: bool | None = None   # None = no holdout run
    overfit: bool = False
    bottleneck_shifted: bool | None = None
    next_action: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GainVerdict:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Candidate:
    """L3: the best config/code found for a target, with its measured Profile."""
    target: str                      # e.g. "config:gpu_memory_utilization"
    kind: str                        # config | code | deploy
    value: dict = field(default_factory=dict)        # the winning knob assignment(s)
    config: dict = field(default_factory=dict)        # full resolved config
    metrics: dict = field(default_factory=dict)       # measured metrics of the winner
    score: float | None = None                        # objective value (per Spec)
    marker_verified: bool = False                     # winner ran real + verified
    evals_used: int = 0
    trials: list = field(default_factory=list)        # [{value, score, verified}] all probes
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Candidate:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Diagnosis:
    """L1: the localized bottleneck + honest confidence."""
    bottleneck: str                  # one of BOTTLENECKS
    status: str                      # "confirmed" | "suspected" | "unknown"
    evidence_refs: list = field(default_factory=list)
    detail: str = ""
    limitations: list = field(default_factory=list)
    ranked: list = field(default_factory=list)   # [{bottleneck, score, status}] runners-up

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Diagnosis:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
