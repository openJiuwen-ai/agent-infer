from __future__ import annotations

import pytest

from vllm_evolve.bench.config import STRONG_BASELINE, build_bench_config
from vllm_evolve.bench.gpu_guard import visible_devices
from vllm_evolve.bench.gpu_selection import (
    bind_auto_gpu_selection,
    select_same_caliber_gpus,
)


def _inventory() -> dict:
    return {
        "captured_at": "2026-07-31T00:00:00+00:00",
        "gpus": [
            {
                "index": str(index),
                "uuid": f"GPU-{index}",
                "name": "NVIDIA L20X",
                "memory_total_mib": 143771,
                "memory_used_mib": used,
            }
            for index, used in enumerate((0, 4, 112872, 0))
        ],
        "compute_apps": [
            {
                "gpu_uuid": "GPU-2",
                "pid": 22104,
                "process_name": "vllm",
                "used_memory_mib": 112000,
            }
        ],
    }


def test_selects_lowest_idle_same_caliber_pair():
    selected = select_same_caliber_gpus(_inventory(), 2)
    assert selected["selected_gpus"] == ["0", "1"]
    gpu2 = next(row for row in selected["examined_gpus"] if row["index"] == "2")
    assert not gpu2["eligible"]
    assert any("compute_apps=22104" in reason for reason in gpu2["excluded_reasons"])


def test_auto_binding_freezes_pair_and_provenance_for_clones():
    config = build_bench_config(
        runner_kind=STRONG_BASELINE,
        gpus="auto",
        tensor_parallel_size=1,
    )
    config.runner.remote = "gpu-box"
    config.workload.workload_spec = {"parallel_mode": "dual_replica"}

    evidence = bind_auto_gpu_selection(config, probe=lambda remote: _inventory())

    assert config.environment.gpus == "0,1"
    assert config.environment.gpu_selection_mode == "auto"
    assert evidence == config.environment.gpu_selection_evidence
    assert evidence["remote"] == "gpu-box"
    clone = type(config).from_dict(config.to_dict())
    assert bind_auto_gpu_selection(
        clone,
        probe=lambda remote: pytest.fail("a frozen clone must not re-probe GPUs"),
    ) is None
    assert clone.environment.gpus == "0,1"
    assert clone.environment.gpu_selection_evidence == evidence


def test_auto_selection_requires_enough_idle_same_caliber_gpus():
    inventory = _inventory()
    inventory["gpus"][1]["name"] = "different-model"
    inventory["gpus"][3]["memory_used_mib"] = 2048
    with pytest.raises(RuntimeError, match="no 2-GPU idle same-caliber set"):
        select_same_caliber_gpus(inventory, 2)


def test_auto_binding_can_wait_for_an_idle_pair():
    config = build_bench_config(
        runner_kind=STRONG_BASELINE,
        gpus="auto",
        tensor_parallel_size=2,
    )
    config.runner.remote = "gpu-box"
    busy = _inventory()
    for gpu in busy["gpus"]:
        gpu["memory_used_mib"] = 120000
    inventories = iter((busy, _inventory()))
    sleeps = []

    evidence = bind_auto_gpu_selection(
        config,
        probe=lambda _remote: next(inventories),
        wait_timeout_s=1.0,
        poll_interval_s=0.1,
        sleep=sleeps.append,
    )

    assert evidence["selected_gpus"] == ["0", "1"]
    assert evidence["selection_attempts"] == 2
    assert sleeps == [0.1]


def test_auto_binding_requires_consecutive_stable_idle_probes():
    config = build_bench_config(
        runner_kind=STRONG_BASELINE,
        gpus="auto",
        tensor_parallel_size=2,
    )
    config.runner.remote = "gpu-box"
    busy = _inventory()
    for gpu in busy["gpus"]:
        gpu["memory_used_mib"] = 120000
    inventories = iter((_inventory(), busy, _inventory(), _inventory()))
    sleeps = []

    evidence = bind_auto_gpu_selection(
        config,
        probe=lambda _remote: next(inventories),
        wait_timeout_s=1.0,
        poll_interval_s=0.1,
        stable_poll_count=2,
        sleep=sleeps.append,
    )

    assert evidence["selected_gpus"] == ["0", "1"]
    assert evidence["selection_attempts"] == 4
    assert evidence["stable_poll_count"] == 2
    assert sleeps == [0.1, 0.1, 0.1]


def test_auto_binding_can_require_server_wide_gpu_service_quiescence():
    config = build_bench_config(
        runner_kind=STRONG_BASELINE,
        gpus="auto",
        tensor_parallel_size=2,
    )
    config.runner.remote = "gpu-box"
    pending = _inventory()
    pending["gpu_service_processes"] = [{
        "user": "peer",
        "pid": 123,
        "command": "vllm serve pending-model",
    }]
    quiet = _inventory()
    quiet["gpu_service_processes"] = []
    inventories = iter((pending, quiet))
    sleeps = []

    evidence = bind_auto_gpu_selection(
        config,
        probe=lambda _remote: next(inventories),
        wait_timeout_s=1.0,
        poll_interval_s=0.1,
        require_server_quiescence=True,
        sleep=sleeps.append,
    )

    assert evidence["selected_gpus"] == ["0", "1"]
    assert evidence["server_quiescence_required"] is True
    assert evidence["selection_attempts"] == 2
    assert sleeps == [0.1]


def test_unresolved_auto_never_reaches_cuda_visible_devices():
    with pytest.raises(ValueError, match="must be resolved"):
        visible_devices("auto")
