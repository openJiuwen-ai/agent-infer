# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Reuse the functional suite's model service within one pytest session."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace

import pytest

from tests.e2e.helpers.case_loader import E2EConfig
from tests.e2e.helpers.server import VllmServer


@pytest.fixture(scope="session")
def agentbench_server() -> Iterator[Callable[[E2EConfig], E2EConfig]]:
    """Own a shared model service for this functional pytest session only."""
    server: VllmServer | None = None

    def use(config: E2EConfig) -> E2EConfig:
        """Bind each benchmark entry to the session's one deployment."""
        nonlocal server
        if server is None:
            server = VllmServer(config=config)
            server.start()
        elif (
            config.case.deploy != server.config.case.deploy
            or config.host != server.config.host
            or config.case.benchmark_params.get("port") != server.config.case.benchmark_params.get("port")
        ):
            raise ValueError("functional benchmarks in one pytest session must share the same deployment")
        if server.proc is None or server.proc.poll() is not None:
            raise RuntimeError("functional vLLM service has exited")
        return replace(
            config,
            host=server.config.host,
            port=server.config.port,
            lifecycle_socket=server.config.lifecycle_socket,
        )

    try:
        yield use
    finally:
        if server is not None:
            server.stop()
