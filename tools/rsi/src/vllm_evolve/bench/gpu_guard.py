"""Hard GPU-budget guard shared by local dispatch and the remote native runner.

The real-hardware workflow is allowed to expose at most two devices to vLLM.
Checking on both sides prevents a malformed CLI/config value from bypassing the
local guard and keeps tensor parallelism inside the visible-device budget.
"""
from __future__ import annotations

MAX_VISIBLE_GPUS = 2


def visible_devices(value: str) -> tuple[str, ...]:
    """Normalize a CUDA device list and enforce the two-device ceiling.

    Numeric indices, GPU UUIDs, and MIG UUIDs are all valid
    ``CUDA_VISIBLE_DEVICES`` tokens, so validation deliberately checks shape
    and cardinality rather than requiring integers.
    """
    if str(value).strip().lower() == "auto":
        raise ValueError(
            "GPU selection 'auto' must be resolved before launching vLLM"
        )
    devices = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not devices:
        raise ValueError("at least one GPU must be selected")
    if len(devices) != len(set(devices)):
        raise ValueError(f"duplicate GPU in selection: {value!r}")
    if len(devices) > MAX_VISIBLE_GPUS:
        raise ValueError(
            f"GPU budget exceeded: selected {len(devices)}, maximum is {MAX_VISIBLE_GPUS}"
        )
    if any(any(ch.isspace() for ch in device) for device in devices):
        raise ValueError(f"invalid whitespace in GPU selection: {value!r}")
    return devices


def validate_gpu_budget(gpus: str, tensor_parallel_size: int | None = 1) -> tuple[str, ...]:
    """Return normalized devices or raise before any vLLM process is started."""
    devices = visible_devices(gpus)
    tp = int(tensor_parallel_size or 1)
    if tp < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if tp > len(devices):
        raise ValueError(
            f"tensor_parallel_size={tp} needs {tp} visible GPUs, only "
            f"{len(devices)} selected"
        )
    if tp > MAX_VISIBLE_GPUS:
        raise ValueError(
            f"tensor_parallel_size={tp} exceeds the {MAX_VISIBLE_GPUS}-GPU budget"
        )
    return devices


def vllm_serve_port_pattern(port: int) -> str:
    """Return an ERE matching a real ``vllm serve`` process on ``port``.

    The command that invokes :mod:`remote_worker` also contains ``vllm`` and a
    ``--port`` argument. A loose pattern therefore kills the worker itself
    before it can persist final status and GPU evidence.
    """
    return (
        rf"(^|/)vllm[[:space:]]+serve.*--port[[:space:]]+{int(port)}"
        r"([[:space:]]|$)"
    )
