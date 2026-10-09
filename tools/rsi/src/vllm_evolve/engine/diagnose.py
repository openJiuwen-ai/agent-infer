"""L1 diagnose — localize the performance bottleneck from a Profile.

A deterministic rule engine (same evidence -> confirmed/suspected/unknown spirit
as ``ops/doctor.py``): each bottleneck has a signature over the profiled signals;
the highest-scoring firing rule wins. **Missing signals never get guessed** — a
rule that needs an absent signal does not fire and the gap is recorded in
``limitations``. If nothing fires, the verdict is ``unknown`` (honest), not a
made-up bottleneck.

Roofline intuition: SM-util high + KV not full => compute-bound; SM-util low +
memory-bandwidth high => bandwidth-bound; GPU idle + short queue => under-saturated.
KV full + preemption => kv_capacity. NCCL-heavy => comm. Capacity free but queue
backed up => scheduling_queue (the only regime where evolving schedule_batch helps).
"""
from __future__ import annotations

from vllm_evolve.core.schemas import (
    COMM,
    COMPUTE,
    KV_CAPACITY,
    MEM_BANDWIDTH,
    PREFILL,
    SCHEDULING_QUEUE,
    UNDER_SATURATED,
    UNKNOWN,
    Diagnosis,
    Profile,
)

# Thresholds (tunable; coarse by design — nvidia-smi/vLLM proxies are not exact).
SM_HIGH = 85.0
SM_LOW = 60.0
SM_IDLE = 40.0
KV_HIGH = 0.90
KV_WARM = 0.80
BW_HIGH = 80.0
NCCL_HIGH = 0.15
NCCL_STRONG = 0.25
PREFILL_HIGH = 0.60
PREFILL_STRONG = 0.75
QUEUE_BACKED = 4  # waiting requests above this == a real queue


def _status(score: float) -> str:
    if score >= 0.85:
        return "confirmed"
    if score >= 0.55:
        return "suspected"
    return "unknown"


def diagnose(profile: Profile | dict) -> Diagnosis:
    if isinstance(profile, dict):
        profile = Profile.from_dict(profile)

    g, v, m = profile.gpu or {}, profile.vllm or {}, profile.metrics or {}
    sm_max = g.get("sm_util_max")
    sm = sm_max if sm_max is not None else g.get("sm_util")  # peak: "ever busy?"
    duty = g.get("duty_cycle")                               # sustained-busy fraction
    bw = g.get("mem_bw_util")
    nccl = g.get("nccl_frac")
    kv = v.get("kv_util")
    preempt = v.get("preempt")
    waiting = v.get("waiting")
    prefill_frac = v.get("prefill_token_frac")

    limitations: list[str] = []
    if sm is None:
        limitations.append("no GPU SM-util — compute/under-saturation rules degraded")
    if duty is None and sm is not None:
        limitations.append("no duty-cycle — can't tell sustained compute from bursty "
                           "idle; compute/under-saturation capped to suspected")
    if bw is None:
        limitations.append("no memory-bandwidth signal — bandwidth-bound rule disabled")
    if kv is None:
        limitations.append("no KV-cache utilization — KV/scheduling rules degraded")
    if not profile.marker_verified:
        limitations.append("profile not marker-verified — metrics may not reflect the candidate")

    # Each rule -> (bottleneck, score, evidence[], detail) or None.
    cands: list[tuple[str, float, list, str]] = []

    # comm (TP/PP) — only when an NCCL fraction was measured.
    if nccl is not None and nccl >= NCCL_HIGH:
        score = 0.9 if nccl >= NCCL_STRONG else 0.6
        cands.append((COMM, score, [f"nccl_frac={nccl:.2f}"],
                      "TP/PP communication is a large share of step time"))

    # kv_capacity — KV full + preemption / deep queue is a strong, specific signal.
    if kv is not None and kv >= KV_WARM:
        if (preempt or 0) > 0 or (waiting or 0) > QUEUE_BACKED:
            score = 0.95 if (kv >= KV_HIGH and (preempt or 0) > 0) else 0.7
            ev = [f"kv_util={kv:.2f}", f"preempt={preempt}", f"waiting={waiting}"]
            cands.append((KV_CAPACITY, score, ev,
                          "KV cache saturated -> preemption / requests queue up"))

    # prefill-bound — COARSE token-rate proxy only -> never above suspected.
    if prefill_frac is not None and prefill_frac >= PREFILL_HIGH:
        ttft = m.get("ttft_p99_ms") or m.get("ttft_mean_ms")
        cands.append((PREFILL, 0.55,
                      [f"prefill_token_frac(proxy)={prefill_frac:.2f}", f"ttft_ms={ttft}"],
                      "high prompt-token share (coarse proxy) -> likely prefill-limited TTFT"))
        limitations.append("prefill signal is a token-RATE proxy, not elapsed prefill time")

    # under-saturated — GPU mostly idle AND queue not backed up (not enough work).
    # High bandwidth => the GPU IS busy (bandwidth-bound), so not idle.
    idle = (duty is not None and duty < 0.5) or (duty is None and sm is not None and sm < SM_LOW)
    if idle and (waiting is None or waiting <= QUEUE_BACKED) \
            and not (bw is not None and bw >= BW_HIGH) \
            and not (kv is not None and kv >= KV_HIGH):
        # confirm only with sustained-idle duty AND a known short queue
        strong = (duty is not None and duty < 0.4) and (waiting is not None)
        cands.append((UNDER_SATURATED, 0.9 if strong else 0.6,
                      [f"duty_cycle={duty}", f"sm_max={sm}", f"waiting={waiting}"],
                      "GPU mostly idle with a short queue -> raise load / admission"))

    # compute-bound — sustained-high SM, KV not the limit, comm low.
    if sm is not None and sm >= SM_HIGH and (duty is None or duty >= 0.7) \
            and not (kv is not None and kv >= KV_HIGH) \
            and not (nccl is not None and nccl >= NCCL_HIGH):
        # CONFIRM only when KV was measured (to rule it out) AND duty proves sustained.
        can_confirm = (kv is not None and duty is not None and duty >= 0.7 and sm >= 92)
        cands.append((COMPUTE, 0.9 if can_confirm else 0.6,
                      [f"sm_max={sm:.0f}", f"duty_cycle={duty}", f"kv_util={kv}"],
                      "GPU FLOPS bound -> quantization / speculative decode / smaller model"))

    # memory-bandwidth-bound — SM not pegged but bandwidth is (coarse proxy).
    if bw is not None and bw >= BW_HIGH and (sm is None or sm < SM_HIGH):
        cands.append((MEM_BANDWIDTH, 0.6,
                      [f"mem_bw_util={bw:.0f}", f"sm_util={sm}"],
                      "byte movement bound -> quantization / larger batch (coarse signal)"))

    # scheduling/queue-bound — capacity exists but the queue is backed up while KV is
    # NOT full and nothing is preempting -> admission/order is the limiter. This is the
    # ONLY regime where evolving schedule_batch helps. "Capacity exists" is evidenced by
    # SM when sampled; with no SM source (e.g. a simulator backend) a measured engine
    # duty-cycle below 0.9 is weaker but real evidence — capped at suspected.
    if waiting is not None and waiting > QUEUE_BACKED \
            and not (kv is not None and kv >= KV_WARM) and (preempt or 0) == 0:
        if sm is not None and SM_LOW <= sm < SM_HIGH:
            cands.append((SCHEDULING_QUEUE, 0.6,
                          [f"waiting={waiting}", f"sm_util={sm:.0f}", f"kv_util={kv}",
                           "preempt=0"],
                          "free capacity but a standing queue -> admission/ordering limits"))
        elif sm is None and duty is not None and duty < 0.9:
            cands.append((SCHEDULING_QUEUE, 0.55,
                          [f"waiting={waiting}", f"duty_cycle={duty}", f"kv_util={kv}",
                           "preempt=0"],
                          "engine not fully busy (duty<0.9) but a standing queue -> "
                          "admission/ordering limits (no SM source; duty evidence)"))

    if not cands:
        return Diagnosis(
            bottleneck=UNKNOWN, status="unknown", evidence_refs=[],
            detail="signals insufficient to localize a bottleneck",
            limitations=limitations or ["no decisive signal in the profile"],
        )

    cands.sort(key=lambda c: c[1], reverse=True)
    top = cands[0]
    ranked = [{"bottleneck": b, "score": round(s, 2), "status": _status(s)}
              for b, s, _, _ in cands]
    return Diagnosis(
        bottleneck=top[0], status=_status(top[1]), evidence_refs=top[2],
        detail=top[3], limitations=limitations, ranked=ranked,
    )
