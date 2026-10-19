"""Real-trace dataset adapters: source format -> canonical TraceDataset.

All adapters normalize to the same shape (see :mod:`.base`). Use
``load_dataset(name, path, **opts)`` to dispatch by name, or import an adapter
module directly.
"""
from __future__ import annotations

from pathlib import Path

from vllm_evolve.bench.datasets import azure, burstgpt, mooncake, sharegpt
from vllm_evolve.bench.datasets.base import (
    TraceDataset,
    TraceRequest,
    char_token_estimate,
)

_LOADERS = {
    "burstgpt": burstgpt.load,
    "azure": azure.load,
    "sharegpt": sharegpt.load,
    "mooncake": mooncake.load,
}


def available() -> list[str]:
    return sorted(_LOADERS)


def load_dataset(name: str, path: str | Path, **opts) -> TraceDataset:
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}; available: {', '.join(available())}")
    return _LOADERS[name](path, **opts)


__all__ = [
    "TraceDataset",
    "TraceRequest",
    "char_token_estimate",
    "available",
    "load_dataset",
]
