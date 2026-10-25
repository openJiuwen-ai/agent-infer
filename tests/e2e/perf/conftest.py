# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Pytest options for E2E benchmark runs."""

from __future__ import annotations

import argparse

import pytest


def _addoption_once(group: pytest.OptionGroup, *args, **kwargs) -> None:
    """Register a CLI option, tolerating duplicate registration.

    tests/e2e/function/tracereplay/conftest.py registers the same shared
    options (--test-config-file / --task-num / --max-concurrency). When both
    conftests are loaded in one session (e.g. ``pytest tests/e2e``), argparse
    raises ``ArgumentError`` for the conflicting option string, which aborts
    the whole collection. The first registration wins; semantics of these
    shared options are identical across both conftests.
    """
    try:
        group.addoption(*args, **kwargs)
    except argparse.ArgumentError:
        pass


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("e2e", "E2E performance benchmark options")
    _addoption_once(
        group,
        "--test-config-file",
        action="store",
        default=None,
        metavar="PATH",
        help=('Case JSON path; scenario from JSON "scenario" (or legacy "arm") or cases/baseline|agentinfer/ path.'),
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
        "--benchmark-run-as-user",
        action="store",
        default=None,
        metavar="USER",
        help="For plan-subagent only: run bench subprocess as USER via sudo when pytest runs as root",
    )
