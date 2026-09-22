from __future__ import annotations

from vllm_evolve.bench.workload_validity import (
    EXTERNAL_CONTAMINATION,
    LOW_KV_OCCUPANCY,
    UNDERLOADED_GPU,
    UNSTABLE_OR_INCOMPLETE,
    VALID,
    evaluate_workload_validity,
)


def _gpu_samples(util: float = 96.0, memory_ratio: float = 0.90):
    rows = []
    for tick in range(20):
        rows.append({
            "captured_at_unix_s": float(tick),
            "gpus": [
                {
                    "index": index,
                    "uuid": f"GPU-{index}",
                    "name": "L20X",
                    "memory_total_mib": 1000,
                    "memory_used_mib": int(1000 * memory_ratio),
                    "utilization_gpu_pct": util,
                    "utilization_memory_pct": 80,
                    "power_draw_w": 300,
                    "clocks_sm_mhz": 1900,
                    "clocks_memory_mhz": 3000,
                    "temperature_gpu_c": 65,
                    "thermal_throttle_active": False,
                    "power_throttle_active": False,
                }
                for index in ("0", "1")
            ],
        })
    return rows


def _vllm_samples(kv: float = 0.92):
    return [
        {
            "captured_at_unix_s": float(tick),
            "metrics": {
                "running_requests": 16,
                "waiting_requests": 8,
                "kv_cache_occupancy": kv,
                "preemptions_total": tick,
                "scheduler_invocations_total": tick * 10,
            },
        }
        for tick in range(20)
    ]


def _evaluate(**overrides):
    values = {
        "gpu_samples": _gpu_samples(),
        "vllm_samples": _vllm_samples(),
        "selected_gpus": ("0", "1"),
        "max_num_seqs": 16,
        "measured_duration_s": 130,
        "measured_requests": 512,
        "completed_requests": 512,
        "error_count": 0,
        "offered_qps": 120,
        "actual_sent_qps": 119,
        "plateau_reference": {
            "offered_qps": 100,
            "achieved_throughput": 90,
            "next_achieved_throughput": 93,
        },
    }
    values.update(overrides)
    return evaluate_workload_validity(**values)


def test_only_fully_saturated_real_run_is_valid():
    result = _evaluate()
    assert result["verdict"] == VALID
    assert result["valid"] is True
    assert result["gpu"]["0"]["utilization_pct"]["p50"] == 96
    assert result["vllm"]["waiting_requests"]["p50"] == 8


def test_software_power_cap_is_recorded_without_invalidating_saturation():
    samples = _gpu_samples()
    for row in samples:
        for gpu in row["gpus"]:
            gpu["power_throttle_active"] = True

    result = _evaluate(gpu_samples=samples)

    assert result["valid"] is True
    assert result["verdict"] == VALID
    assert result["gpu"]["0"]["power_throttle_seen"] is True
    assert result["gpu"]["1"]["power_throttle_seen"] is True


def test_thermal_or_hardware_power_brake_still_invalidates_measurement():
    samples = _gpu_samples()
    samples[0]["gpus"][0]["thermal_throttle_active"] = True

    result = _evaluate(gpu_samples=samples)

    assert result["verdict"] == UNSTABLE_OR_INCOMPLETE
    assert any("hardware power braking" in reason for reason in result["reasons"])


def test_one_transient_memory_spike_is_recorded_without_invalidating_window():
    samples = _gpu_samples()
    samples[0]["gpus"][0]["memory_used_mib"] = 990

    result = _evaluate(gpu_samples=samples)

    assert result["valid"] is True
    assert result["gpu"]["0"]["memory_ratio"]["p95"] == 0.90
    assert result["gpu"]["0"]["memory_ratio"]["max"] == 0.99


def test_high_p95_allocator_retention_is_recorded_without_invalidating_window():
    samples = _gpu_samples()
    samples[0]["gpus"][0]["memory_used_mib"] = 990
    samples[1]["gpus"][0]["memory_used_mib"] = 990

    result = _evaluate(gpu_samples=samples)

    assert result["valid"] is True
    assert result["gpu"]["0"]["memory_ratio"]["p95"] == 0.99
    assert result["gpu"]["0"]["memory_ratio"]["max"] == 0.99


def test_frozen_explicit_plateau_pair_is_reusable_by_every_scenario():
    pair = {
        "base_offered_qps": 100,
        "higher_offered_qps": 120,
        "base_achieved_throughput": 90,
        "higher_achieved_throughput": 93,
    }
    saturated = _evaluate(
        offered_qps=100,
        actual_sent_qps=99,
        plateau_reference=pair,
    )
    severe = _evaluate(
        offered_qps=500,
        actual_sent_qps=499,
        plateau_reference=pair,
    )
    assert saturated["valid"] is True
    assert severe["valid"] is True
    assert saturated["plateau"]["proof_kind"] == (
        "frozen_explicit_calibration_pair"
    )
    assert saturated["plateau"] == severe["plateau"]


def test_allocator_reservation_without_live_kv_pressure_is_invalid():
    result = _evaluate(vllm_samples=_vllm_samples(kv=0.10))
    assert result["verdict"] == LOW_KV_OCCUPANCY
    assert result["valid"] is False


def test_low_compute_is_invalid_even_when_memory_is_full():
    result = _evaluate(gpu_samples=_gpu_samples(util=40))
    assert result["verdict"] == UNDERLOADED_GPU


def test_decode_bound_gpu_utilization_is_valid_with_full_scheduler_pressure():
    samples = _gpu_samples(util=80)
    for row in samples[:2]:
        for gpu in row["gpus"]:
            gpu["utilization_gpu_pct"] = 50
    for row in samples[-2:]:
        for gpu in row["gpus"]:
            gpu["utilization_gpu_pct"] = 100

    result = _evaluate(gpu_samples=samples)

    assert result["valid"] is True
    assert result["gpu"]["0"]["utilization_pct"]["p50"] == 80
    assert result["vllm"]["kv_cache_occupancy"]["p50"] == 0.92


def test_foreign_process_has_highest_precedence():
    result = _evaluate(
        contaminated=True,
        gpu_samples=_gpu_samples(util=40),
        vllm_samples=_vllm_samples(kv=0.1),
    )
    assert result["verdict"] == EXTERNAL_CONTAMINATION


def test_formal_pair_must_report_same_gpu_model_and_capacity():
    samples = _gpu_samples()
    for row in samples:
        row["gpus"][1]["name"] = "different-model"

    result = _evaluate(gpu_samples=samples)

    assert result["verdict"] == UNSTABLE_OR_INCOMPLETE
    assert any("same reported model" in reason for reason in result["reasons"])


def test_actual_measured_window_still_enforces_120_seconds():
    result = _evaluate(measured_duration_s=119.9)

    assert result["verdict"] == UNSTABLE_OR_INCOMPLETE
    assert any("measured duration" in reason for reason in result["reasons"])
