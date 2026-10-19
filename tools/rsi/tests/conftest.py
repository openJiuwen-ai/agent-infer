"""Shared pytest configuration.

Registers the ``requires_gpu`` marker for tests that need a real GPU + Docker +
the pinned vLLM image. These are skipped by default; set ``VLLM_EVOLVE_GPU=1``
on a GPU host to run them. This is how GPU-gated code stays honest: the tests
exist and run on hardware, but never fake a pass off-hardware.
"""
from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_gpu: needs a real GPU + Docker + pinned vLLM image "
        "(set VLLM_EVOLVE_GPU=1 to run)",
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("VLLM_EVOLVE_GPU") == "1":
        return
    skip = pytest.mark.skip(reason="GPU-gated; set VLLM_EVOLVE_GPU=1 on a GPU host to run")
    for item in items:
        if "requires_gpu" in item.keywords:
            item.add_marker(skip)
