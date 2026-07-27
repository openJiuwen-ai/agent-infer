# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentCache project
"""Tests for additive AgentInfer logging inside a vLLM process."""

import json
import logging
import subprocess
import sys

import pytest

from agentinfer.agentcache.core.vllm_logging import attach_agentinfer_to_vllm_logging

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def isolated_runtime_loggers():
    """Restore logger state after exercising process-global logging objects."""
    names = ("agentinfer", "vllm", "uvicorn.error", "uvicorn.access")
    loggers = {name: logging.getLogger(name) for name in names}
    saved = {name: (list(logger.handlers), logger.level, logger.propagate) for name, logger in loggers.items()}
    try:
        for logger in loggers.values():
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
        yield loggers
    finally:
        for name, logger in loggers.items():
            handlers, level, propagate = saved[name]
            logger.handlers[:] = handlers
            logger.setLevel(level)
            logger.propagate = propagate


def test_attach_reuses_vllm_handler_without_touching_uvicorn(isolated_runtime_loggers) -> None:
    """AgentInfer should inherit formatting while Uvicorn keeps native handlers."""
    loggers = isolated_runtime_loggers
    vllm_handler = logging.StreamHandler()
    vllm_handler.setFormatter(logging.Formatter("vllm:%(message)s"))
    uvicorn_error_handler = logging.StreamHandler()
    uvicorn_access_handler = logging.StreamHandler()
    loggers["vllm"].addHandler(vllm_handler)
    loggers["vllm"].setLevel(logging.INFO)
    loggers["uvicorn.error"].addHandler(uvicorn_error_handler)
    loggers["uvicorn.access"].addHandler(uvicorn_access_handler)

    attach_agentinfer_to_vllm_logging()
    attach_agentinfer_to_vllm_logging()

    assert loggers["agentinfer"].handlers == [vllm_handler]
    assert loggers["agentinfer"].level == logging.INFO
    assert loggers["agentinfer"].propagate is False
    assert loggers["uvicorn.error"].handlers == [uvicorn_error_handler]
    assert loggers["uvicorn.access"].handlers == [uvicorn_access_handler]


def test_attach_preserves_explicit_agentinfer_logging(isolated_runtime_loggers) -> None:
    """A user-owned AgentInfer handler must take precedence over inheritance."""
    loggers = isolated_runtime_loggers
    vllm_handler = logging.StreamHandler()
    explicit_handler = logging.StreamHandler()
    loggers["vllm"].addHandler(vllm_handler)
    loggers["agentinfer"].addHandler(explicit_handler)
    loggers["agentinfer"].setLevel(logging.DEBUG)

    attach_agentinfer_to_vllm_logging()

    assert loggers["agentinfer"].handlers == [explicit_handler]
    assert loggers["agentinfer"].level == logging.DEBUG
    assert loggers["agentinfer"].propagate is True


def test_attach_waits_for_vllm_logging_initialization(isolated_runtime_loggers) -> None:
    """An early import must remain a no-op until vLLM has configured handlers."""
    loggers = isolated_runtime_loggers

    attach_agentinfer_to_vllm_logging()

    assert loggers["agentinfer"].handlers == []
    assert loggers["agentinfer"].level == logging.NOTSET
    assert loggers["agentinfer"].propagate is True


def test_vllm_integration_module_import_does_not_attach_logging() -> None:
    """Importing a type-only integration module must not mutate process logging."""
    script = """
import json
import logging

agentinfer_logger = logging.getLogger("agentinfer")
vllm_logger = logging.getLogger("vllm")
vllm_logger.addHandler(logging.StreamHandler())
vllm_logger.setLevel(logging.INFO)
import agentinfer.agentcache.core.api_adapter
import agentinfer.agentcache.core.scheduler
print("RESULT=" + json.dumps({
    "handlers": len(agentinfer_logger.handlers),
    "level": agentinfer_logger.level,
    "propagate": agentinfer_logger.propagate,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result_line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULT="))
    result = json.loads(result_line.removeprefix("RESULT="))

    assert result == {
        "handlers": 0,
        "level": logging.NOTSET,
        "propagate": True,
    }
