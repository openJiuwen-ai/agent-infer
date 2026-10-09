from __future__ import annotations

import pytest

from vllm_evolve.bench.gpu_guard import (
    validate_gpu_budget,
    visible_devices,
    vllm_serve_port_pattern,
)


def test_one_or_two_devices_are_allowed():
    assert visible_devices("1") == ("1",)
    assert visible_devices("1,3") == ("1", "3")
    assert visible_devices("GPU-a,GPU-b") == ("GPU-a", "GPU-b")


@pytest.mark.parametrize("value", ["", ",", "0,1,2", "1,1", "GPU-a, GPU-a"])
def test_empty_duplicate_or_more_than_two_is_rejected(value):
    with pytest.raises(ValueError):
        visible_devices(value)


def test_tensor_parallelism_must_fit_visible_devices_and_budget():
    assert validate_gpu_budget("1,3", 2) == ("1", "3")
    with pytest.raises(ValueError, match="only 1 selected"):
        validate_gpu_budget("1", 2)
    with pytest.raises(ValueError, match="maximum is 2"):
        validate_gpu_budget("0,1,2", 3)


def test_cleanup_pattern_only_matches_real_vllm_serve_process():
    pattern = vllm_serve_port_pattern(8260)
    assert "vllm[[:space:]]+serve" in pattern
    assert "vllm.*--port" not in pattern
    assert "--port[[:space:]]+8260" in pattern
