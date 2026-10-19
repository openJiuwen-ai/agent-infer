"""Measured-window sampler for the live vLLM Prometheus endpoint."""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from pathlib import Path

_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s+\d+)?$"
)

_ALIASES = {
    "running_requests": (
        "vllm:num_requests_running",
        "vllm_num_requests_running",
    ),
    "waiting_requests": (
        "vllm:num_requests_waiting",
        "vllm_num_requests_waiting",
    ),
    "kv_cache_occupancy": (
        "vllm:kv_cache_usage_perc",
        "vllm_kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "vllm_gpu_cache_usage_perc",
    ),
    "preemptions_total": (
        "vllm:num_preemptions_total",
        "vllm_num_preemptions_total",
    ),
    "prompt_tokens_total": (
        "vllm:prompt_tokens_total",
        "vllm_prompt_tokens_total",
    ),
    "generation_tokens_total": (
        "vllm:generation_tokens_total",
        "vllm_generation_tokens_total",
    ),
    # Prometheus histogram count equals completed engine iterations. In V1,
    # every engine iteration is driven by one scheduler decision, so its
    # measured-window delta is the non-invasive baseline/candidate-compatible
    # scheduler invocation counter.
    "scheduler_invocations_total": (
        "vllm:iteration_tokens_total_count",
        "vllm_iteration_tokens_total_count",
    ),
}


def parse_prometheus_metrics(text: str) -> dict[str, float | None]:
    """Extract scheduler pressure fields across vLLM metric-name versions."""
    raw: dict[str, list[float]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line.strip())
        if not match:
            continue
        raw.setdefault(match.group("name"), []).append(float(match.group("value")))

    result: dict[str, float | None] = {}
    for canonical, aliases in _ALIASES.items():
        values = [
            value
            for alias in aliases
            for value in raw.get(alias, [])
        ]
        if not values:
            result[canonical] = None
        elif canonical == "kv_cache_occupancy":
            # Multiple cache groups can be exported; the busiest group is the
            # conservative scheduler-pressure signal and must never be summed.
            result[canonical] = max(values)
        else:
            result[canonical] = sum(values)
    return result


def fetch_vllm_metrics(port: int, *, timeout_s: float = 5.0) -> dict:
    with urllib.request.urlopen(  # noqa: S310 - fixed localhost endpoint
        f"http://127.0.0.1:{int(port)}/metrics",
        timeout=timeout_s,
    ) as response:
        text = response.read().decode("utf-8", errors="replace")
    return parse_prometheus_metrics(text)


class VLLMMetricsSampler:
    """Append live metrics only while one measured seed is in progress."""

    def __init__(
        self,
        *,
        port: int,
        path: str | Path,
        seed: int,
        interval_s: float = 1.0,
    ):
        self.port = int(port)
        self.path = Path(path)
        self.seed = int(seed)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            while True:
                row = {
                    "captured_at_unix_s": time.time(),
                    "seed": self.seed,
                    "phase": "measured",
                    "metrics": {},
                    "error": None,
                }
                try:
                    row["metrics"] = fetch_vllm_metrics(self.port)
                except Exception as exc:  # evidence gap, never fabricated pressure
                    row["error"] = f"{type(exc).__name__}: {exc}"
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                if self._stop.wait(self.interval_s):
                    break

    def start(self) -> VLLMMetricsSampler:
        if self._thread is not None:
            raise RuntimeError("vLLM metrics sampler already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self.interval_s * 3))

    def __enter__(self) -> VLLMMetricsSampler:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
