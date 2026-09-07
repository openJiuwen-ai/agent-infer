# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Candidate Router session registration and cleanup."""

import time
from dataclasses import dataclass
from typing import Literal

import httpx


@dataclass(frozen=True)
class SessionRegistrationResult:
    operation: Literal["register", "cleanup"]
    session_id: str
    success: bool
    status_code: int | None
    latency_seconds: float
    error: str | None


async def _operate(
    router_url: str, operation: Literal["register", "cleanup"], session_id: str, timeout: float
) -> SessionRegistrationResult:
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(
                f"{router_url.rstrip('/')}/v1/sessions/{operation}", json={"session_id": session_id}
            )
        return SessionRegistrationResult(
            operation,
            session_id,
            response.is_success,
            response.status_code,
            time.monotonic() - started,
            None if response.is_success else response.text,
        )
    except httpx.HTTPError as exc:
        return SessionRegistrationResult(operation, session_id, False, None, time.monotonic() - started, str(exc))


async def register_session(router_url: str, session_id: str, timeout_seconds: float = 10) -> SessionRegistrationResult:
    return await _operate(router_url, "register", session_id, timeout_seconds)


async def cleanup_session(router_url: str, session_id: str, timeout_seconds: float = 10) -> SessionRegistrationResult:
    return await _operate(router_url, "cleanup", session_id, timeout_seconds)
