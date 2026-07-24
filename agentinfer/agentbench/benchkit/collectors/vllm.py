"""Capture raw vLLM Prometheus evidence without writing artifacts."""

import httpx

from ..metrics.schema import EvidenceCapture


async def capture_vllm_metrics(base_url: str) -> EvidenceCapture:
    """Fetch raw vLLM Prometheus text without parsing or aggregation."""

    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            response = await client.get(f"{base_url.rstrip('/')}/metrics")
            response.raise_for_status()
        return EvidenceCapture("vllm", None, True, None, {"text": response.text})
    except httpx.HTTPError as exc:
        return EvidenceCapture("vllm", None, False, str(exc), {})
