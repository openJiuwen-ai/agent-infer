# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Capture raw Router evidence without aggregating or writing artifacts."""

import httpx

from ..metrics.schema import EvidenceCapture


async def capture_router_snapshot(router_url: str) -> EvidenceCapture:
    """Fetch the Router metrics payload without aggregating its events."""

    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            response = await client.get(f"{router_url.rstrip('/')}/metrics")
            response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, (dict, list)):
            return EvidenceCapture("router", None, False, "metrics must contain a JSON object or array", {})
        return EvidenceCapture("router", None, True, None, {"raw": payload})
    except (httpx.HTTPError, ValueError) as exc:
        return EvidenceCapture("router", None, False, str(exc), {})
