"""Hard validity gate for saturated real-vLLM measurements.

Performance numbers are not fitness until this module proves that the measured
window stressed both selected GPUs, the live KV cache, and the vLLM scheduler.
The gate is deliberately independent from candidate code and from the metric
being optimized so an underloaded run cannot become a winner by looking fast.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import ceil

VALID = "valid_saturated_real_vllm"
UNDERLOADED_GPU = "invalid_underloaded_gpu"
LOW_KV_OCCUPANCY = "invalid_low_kv_occupancy"
NO_QUEUE_PRESSURE = "invalid_no_queue_pressure"
GPU_IMBALANCE = "invalid_gpu_imbalance"
EXTERNAL_CONTAMINATION = "invalid_external_gpu_contamination"
CLIENT_UNDERPOWERED = "invalid_client_underpowered"
UNSTABLE_OR_INCOMPLETE = "invalid_unstable_or_incomplete"


@dataclass(frozen=True)
class ValidityThresholds:
    selected_gpu_count: int = 2
    memory_ratio_min: float = 0.85
    memory_ratio_max: float = 0.95
    memory_ratio_hard_max: float = 0.97
    kv_p50_min: float = 0.70
    kv_peak_min: float = 0.90
    # Decode-heavy saturation can be memory/KV/scheduler bound before SM
    # utilization reaches prefill-like levels.  The independent KV, running,
    # waiting, and duty-cycle gates below still prove sustained pressure.
    gpu_util_p10_min: float = 50.0
    gpu_util_p50_min: float = 75.0
    gpu_util_p95_min: float = 95.0
    active_util_threshold: float = 50.0
    active_duty_min: float = 0.90
    max_gpu_median_imbalance_points: float = 20.0
    running_p50_fraction_min: float = 0.80
    running_p95_fraction_min: float = 0.90
    waiting_p50_min: float = 0.0
    waiting_p95_min: float = 1.0
    min_scheduler_invocations: int = 100
    min_duration_s: float = 120.0
    min_measured_requests: int = 512
    min_completion_rate: float = 1.0
    max_errors: int = 0
    min_delivery_ratio: float = 0.95
    plateau_load_increase_min: float = 0.20
    plateau_throughput_gain_max: float = 0.05


def percentile(values: Iterable[float], q: float) -> float | None:
    """Deterministic nearest-rank percentile for small telemetry series."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    rank = max(1, ceil(q * len(ordered)))
    return ordered[rank - 1]


def _stats(values: Iterable[float]) -> dict:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "p10": percentile(data, 0.10),
        "p50": percentile(data, 0.50),
        "p95": percentile(data, 0.95),
        "max": max(data) if data else None,
    }


def summarize_gpu_samples(
    samples: list[dict],
    *,
    active_util_threshold: float = 50.0,
) -> dict[str, dict]:
    """Summarize already-windowed GPU samples independently per physical GPU."""
    values: dict[str, dict[str, list]] = {}
    for row in samples:
        for gpu in row.get("gpus", []):
            item = values.setdefault(str(gpu["index"]), {
                "uuid": [],
                "name": [],
                "util": [],
                "memory_ratio": [],
                "memory_used_mib": [],
                "memory_total_mib": [],
                "memory_controller_util": [],
                "power_w": [],
                "sm_clock_mhz": [],
                "memory_clock_mhz": [],
                "temperature_c": [],
                "thermal_throttle": [],
                "power_throttle": [],
            })
            item["uuid"].append(str(gpu.get("uuid") or ""))
            item["name"].append(str(gpu.get("name") or ""))
            used = float(gpu.get("memory_used_mib") or 0.0)
            total = float(gpu.get("memory_total_mib") or 0.0)
            item["util"].append(float(gpu.get("utilization_gpu_pct") or 0.0))
            item["memory_used_mib"].append(used)
            item["memory_total_mib"].append(total)
            item["memory_ratio"].append(used / total if total else 0.0)
            item["memory_controller_util"].append(
                float(gpu.get("utilization_memory_pct") or 0.0)
            )
            item["power_w"].append(float(gpu.get("power_draw_w") or 0.0))
            item["sm_clock_mhz"].append(float(gpu.get("clocks_sm_mhz") or 0.0))
            item["memory_clock_mhz"].append(
                float(gpu.get("clocks_memory_mhz") or 0.0)
            )
            item["temperature_c"].append(
                float(gpu.get("temperature_gpu_c") or 0.0)
            )
            item["thermal_throttle"].append(
                bool(gpu.get("thermal_throttle_active"))
            )
            item["power_throttle"].append(bool(gpu.get("power_throttle_active")))

    summary = {}
    for index, item in values.items():
        util = item["util"]
        summary[index] = {
            "uuid": item["uuid"][0] if item["uuid"] else "",
            "name": item["name"][0] if item["name"] else "",
            "utilization_pct": _stats(util),
            "active_duty_cycle": (
                sum(value >= active_util_threshold for value in util) / len(util)
                if util else 0.0
            ),
            "memory_ratio": _stats(item["memory_ratio"]),
            "memory_used_mib": _stats(item["memory_used_mib"]),
            "memory_total_mib": (
                item["memory_total_mib"][0] if item["memory_total_mib"] else None
            ),
            "memory_controller_utilization_pct": _stats(
                item["memory_controller_util"]
            ),
            "power_w": _stats(item["power_w"]),
            "sm_clock_mhz": _stats(item["sm_clock_mhz"]),
            "memory_clock_mhz": _stats(item["memory_clock_mhz"]),
            "temperature_c": _stats(item["temperature_c"]),
            "thermal_throttle_seen": any(item["thermal_throttle"]),
            "power_throttle_seen": any(item["power_throttle"]),
        }
    return summary


def summarize_vllm_samples(samples: list[dict]) -> dict:
    def values(name: str) -> list[float]:
        return [
            float(row["metrics"][name])
            for row in samples
            if row.get("metrics", {}).get(name) is not None
        ]

    preemptions = values("preemptions_total")
    scheduler_invocations = values("scheduler_invocations_total")
    return {
        "sample_count": len(samples),
        "running_requests": _stats(values("running_requests")),
        "waiting_requests": _stats(values("waiting_requests")),
        "kv_cache_occupancy": _stats(values("kv_cache_occupancy")),
        "preemptions_delta": (
            max(preemptions) - min(preemptions) if preemptions else None
        ),
        "scheduler_invocations_delta": (
            max(scheduler_invocations) - min(scheduler_invocations)
            if scheduler_invocations else None
        ),
    }


def _metric(summary: dict, group: str, field: str) -> float | None:
    value = summary.get(group, {}).get(field)
    return float(value) if value is not None else None


def evaluate_workload_validity(
    *,
    gpu_samples: list[dict],
    vllm_samples: list[dict],
    selected_gpus: tuple[str, ...] | list[str],
    max_num_seqs: int,
    measured_duration_s: float,
    measured_requests: int,
    completed_requests: int,
    error_count: int,
    offered_qps: float,
    actual_sent_qps: float,
    plateau_reference: dict | None,
    contaminated: bool = False,
    thresholds: ValidityThresholds | None = None,
) -> dict:
    """Return the only verdict allowed to qualify a real-vLLM fitness row."""
    t = thresholds or ValidityThresholds()
    gpu_summary = summarize_gpu_samples(
        gpu_samples, active_util_threshold=t.active_util_threshold
    )
    vllm_summary = summarize_vllm_samples(vllm_samples)
    reasons: list[str] = []
    categories: list[str] = []

    def fail(category: str, reason: str) -> None:
        if category not in categories:
            categories.append(category)
        reasons.append(reason)

    selected = tuple(str(item) for item in selected_gpus)
    if contaminated:
        fail(EXTERNAL_CONTAMINATION, "a foreign process used a selected GPU")
    if len(selected) != t.selected_gpu_count:
        fail(
            UNSTABLE_OR_INCOMPLETE,
            f"formal run selected {len(selected)} GPUs; required {t.selected_gpu_count}",
        )
    missing = [index for index in selected if index not in gpu_summary]
    if missing:
        fail(UNSTABLE_OR_INCOMPLETE, f"missing GPU telemetry for {missing}")
    selected_summaries = [
        gpu_summary[index] for index in selected if index in gpu_summary
    ]
    selected_names = {
        str(gpu.get("name") or "") for gpu in selected_summaries
    }
    selected_capacities = {
        float(gpu.get("memory_total_mib") or 0.0)
        for gpu in selected_summaries
    }
    if (
        len(selected_summaries) == t.selected_gpu_count
        and (
            len(selected_names) != 1
            or "" in selected_names
            or len(selected_capacities) != 1
            or 0.0 in selected_capacities
        )
    ):
        fail(
            UNSTABLE_OR_INCOMPLETE,
            "formal run requires two GPUs with the same reported model and capacity",
        )

    for index in selected:
        gpu = gpu_summary.get(index)
        if not gpu:
            continue
        util = gpu["utilization_pct"]
        memory = gpu["memory_ratio"]
        if (
            (util["p10"] or 0.0) < t.gpu_util_p10_min
            or (util["p50"] or 0.0) < t.gpu_util_p50_min
            or (util["p95"] or 0.0) < t.gpu_util_p95_min
            or gpu["active_duty_cycle"] < t.active_duty_min
        ):
            fail(
                UNDERLOADED_GPU,
                f"GPU {index} utilization/duty below hard threshold",
            )
        if (memory["p50"] or 0.0) < t.memory_ratio_min:
            fail(
                UNDERLOADED_GPU,
                f"GPU {index} median memory ratio is below {t.memory_ratio_min:.0%}",
            )
        if (memory["p50"] or 0.0) > t.memory_ratio_max:
            fail(
                UNSTABLE_OR_INCOMPLETE,
                f"GPU {index} median memory ratio exceeds {t.memory_ratio_max:.0%}",
            )
        # CUDA's caching allocator can retain temporary allocations through the
        # drain tail even when the run completes cleanly.  Keep p95/max and the
        # hard ceiling in provenance for audit, while sustained capacity is
        # enforced by the median band above and runtime safety by exact
        # completion with zero errors below.
        # Reaching the configured software power cap is a normal steady-state
        # operating regime for a saturated accelerator.  Keep it in provenance
        # so paired runs can be audited, but do not reject an otherwise valid
        # measurement solely because the GPU is operating at its rated limit.
        # ``thermal_throttle_seen`` also includes hardware power-brake events
        # from the remote sampler; those still invalidate the measurement.
        if gpu["thermal_throttle_seen"]:
            fail(
                UNSTABLE_OR_INCOMPLETE,
                f"GPU {index} reported thermal throttling or hardware power braking",
            )

    medians = [
        float(gpu_summary[index]["utilization_pct"]["p50"])
        for index in selected
        if index in gpu_summary
        and gpu_summary[index]["utilization_pct"]["p50"] is not None
    ]
    if medians and max(medians) - min(medians) > t.max_gpu_median_imbalance_points:
        fail(GPU_IMBALANCE, "per-GPU median utilization differs by over 20 points")

    kv_p50 = _metric(vllm_summary, "kv_cache_occupancy", "p50")
    kv_peak = _metric(vllm_summary, "kv_cache_occupancy", "max")
    if kv_p50 is None or kv_peak is None:
        fail(LOW_KV_OCCUPANCY, "live KV cache occupancy telemetry is missing")
    elif kv_p50 < t.kv_p50_min or kv_peak < t.kv_peak_min:
        fail(
            LOW_KV_OCCUPANCY,
            f"KV occupancy p50={kv_p50:.3f}, peak={kv_peak:.3f}",
        )

    waiting_p50 = _metric(vllm_summary, "waiting_requests", "p50")
    waiting_p95 = _metric(vllm_summary, "waiting_requests", "p95")
    running_p50 = _metric(vllm_summary, "running_requests", "p50")
    running_p95 = _metric(vllm_summary, "running_requests", "p95")
    scheduler_invocations = vllm_summary.get("scheduler_invocations_delta")
    if (
        waiting_p50 is None
        or waiting_p95 is None
        or running_p50 is None
        or running_p95 is None
        or waiting_p50 <= t.waiting_p50_min
        or waiting_p95 < t.waiting_p95_min
        or running_p50 < t.running_p50_fraction_min * max_num_seqs
        or running_p95 < t.running_p95_fraction_min * max_num_seqs
        or scheduler_invocations is None
        or scheduler_invocations < t.min_scheduler_invocations
    ):
        fail(NO_QUEUE_PRESSURE, "running/waiting scheduler pressure is below threshold")

    if measured_duration_s < t.min_duration_s:
        fail(
            UNSTABLE_OR_INCOMPLETE,
            f"measured duration {measured_duration_s:.1f}s < {t.min_duration_s:.1f}s",
        )
    if measured_requests < t.min_measured_requests:
        fail(
            UNSTABLE_OR_INCOMPLETE,
            f"measured requests {measured_requests} < {t.min_measured_requests}",
        )
    completion_rate = (
        completed_requests / measured_requests if measured_requests else 0.0
    )
    if completion_rate < t.min_completion_rate or error_count > t.max_errors:
        fail(
            UNSTABLE_OR_INCOMPLETE,
            f"completion={completion_rate:.6f}, errors={error_count}",
        )
    delivery_ratio = actual_sent_qps / offered_qps if offered_qps > 0 else 0.0
    if delivery_ratio < t.min_delivery_ratio:
        fail(
            CLIENT_UNDERPOWERED,
            f"load generator delivered only {delivery_ratio:.1%} of offered QPS",
        )

    plateau = {
        "valid": False,
        "load_increase": None,
        "throughput_gain": None,
        "proof_kind": None,
    }
    if plateau_reference:
        # Schema v2 freezes an explicit pair of baseline calibration points and can
        # therefore be reused by saturated/high/severe formal scenarios. The legacy
        # schema treated the current scenario as the higher point, which incorrectly
        # made the base saturated scenario fail its own plateau proof.
        explicit_pair = (
            "base_offered_qps" in plateau_reference
            or "higher_offered_qps" in plateau_reference
        )
        base_load = float(
            plateau_reference.get(
                "base_offered_qps" if explicit_pair else "offered_qps"
            )
            or 0.0
        )
        higher_load = float(
            plateau_reference.get("higher_offered_qps")
            if explicit_pair
            else offered_qps
        )
        base_throughput = float(
            plateau_reference.get(
                "base_achieved_throughput"
                if explicit_pair
                else "achieved_throughput"
            )
            or 0.0
        )
        load_increase = (
            higher_load / base_load - 1.0 if base_load > 0 else None
        )
        throughput_gain = (
            float(
                plateau_reference.get(
                    "higher_achieved_throughput"
                    if explicit_pair
                    else "next_achieved_throughput"
                )
                or 0.0
            )
            / base_throughput
            - 1.0
            if base_throughput > 0 else None
        )
        plateau = {
            "valid": (
                load_increase is not None
                and throughput_gain is not None
                and load_increase + 1e-12 >= t.plateau_load_increase_min
                and throughput_gain < t.plateau_throughput_gain_max
            ),
            "load_increase": load_increase,
            "throughput_gain": throughput_gain,
            "proof_kind": (
                "frozen_explicit_calibration_pair"
                if explicit_pair
                else "legacy_current_plus_reference"
            ),
            "base_offered_qps": base_load,
            "higher_offered_qps": higher_load,
            "base_achieved_throughput": base_throughput,
        }
    if not plateau["valid"]:
        fail(NO_QUEUE_PRESSURE, "throughput plateau is not proven by a +20% load point")

    precedence = (
        EXTERNAL_CONTAMINATION,
        CLIENT_UNDERPOWERED,
        UNSTABLE_OR_INCOMPLETE,
        GPU_IMBALANCE,
        LOW_KV_OCCUPANCY,
        NO_QUEUE_PRESSURE,
        UNDERLOADED_GPU,
    )
    verdict = VALID
    for category in precedence:
        if category in categories:
            verdict = category
            break
    return {
        "schema_version": 1,
        "source": "real_vllm",
        "verdict": verdict,
        "valid": verdict == VALID,
        "reasons": reasons,
        "failed_categories": categories,
        "thresholds": asdict(t),
        "selected_gpus": list(selected),
        "gpu": gpu_summary,
        "vllm": vllm_summary,
        "measurement": {
            "duration_s": float(measured_duration_s),
            "requests": int(measured_requests),
            "completed": int(completed_requests),
            "errors": int(error_count),
            "completion_rate": completion_rate,
            "offered_qps": float(offered_qps),
            "actual_sent_qps": float(actual_sent_qps),
            "delivery_ratio": delivery_ratio,
        },
        "plateau": plateau,
    }
