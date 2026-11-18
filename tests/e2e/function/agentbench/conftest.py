# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Reuse the functional suite's model service within one pytest session."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import replace

import pytest

from tests.e2e.helpers.case_loader import E2EConfig
from tests.e2e.helpers.server import VllmServer


@pytest.fixture(scope="session")
def agentbench_server() -> Iterator[Callable[[E2EConfig], E2EConfig]]:
    server: VllmServer | None = None
    signature: str | None = None

    def use(config: E2EConfig) -> E2EConfig:
        nonlocal server, signature
        spec = config.case.deploy
        key = json.dumps(
            {
                "model": str(spec.model),
                "args": spec.serve_args,
                "env": spec.serve_env,
                "middleware": spec.middleware,
                "scenario": config.scenario.value,
                "host": config.host,
                "port": config.case.benchmark_params.get("port"),
            },
            sort_keys=True,
        )
        if server is None or key != signature:
            if server is not None:
                server.stop()
            server = VllmServer(config=config)
            try:
                server.start()
            except BaseException:
                server.stop()
                server = None
                raise
            signature = key
        if server.proc is None or server.proc.poll() is not None:
            raise RuntimeError("shared functional vLLM service has exited")
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
