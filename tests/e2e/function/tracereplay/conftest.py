# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Shared pytest options for Trace Replay functional E2E."""

from __future__ import annotations

import argparse

import pytest


def _addoption_once(group: pytest.OptionGroup, *args, **kwargs) -> None:
    """Register a CLI option, tolerating duplicate registration.

    tests/e2e/perf/conftest.py registers the same shared options
    (--test-config-file / --task-num / --max-concurrency). When both conftests
    are loaded in one session (e.g. ``pytest tests/e2e``), argparse raises
    ``ArgumentError`` for the conflicting option string, which aborts the whole
    collection. The first registration wins; semantics of these shared options
    are identical across both conftests.
    """
    try:
        group.addoption(*args, **kwargs)
    except argparse.ArgumentError:
        pass


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup(
        "tracereplay_e2e",
        "Trace Replay functional E2E options (inferact, tracelab, agentx)",
    )
    _addoption_once(
        group,
        "--test-config-file",
        action="store",
        default=None,
        metavar="PATH",
        help="Replay case JSON under tests/e2e/function/tracereplay/cases/<suite>/",
    )
    _addoption_once(
        group,
        "--task-num",
        action="store",
        type=int,
        default=None,
        help="Override benchmark_params task-num",
    )
    _addoption_once(
        group,
        "--max-concurrency",
        action="store",
        type=int,
        default=None,
        help="Override benchmark_params max-concurrency",
    )
    group.addoption(
        "--bench-config",
        action="store",
        default=None,
        metavar="PATH",
        help="Override benchmark_params config (replay bench YAML for --config)",
    )
    group.addoption(
        "--cold-runs",
        action="store",
        type=int,
        default=2,
        help="Number of cold vLLM + replay cycles (default: 2; compare uses runs 1 and 2)",
    )
    group.addoption(
        "--stability-tolerance",
        action="store",
        type=float,
        default=0.05,
        help="Maximum relative metric gap for repeatability checks (default: 0.05)",
    )
