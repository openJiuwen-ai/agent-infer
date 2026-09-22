"""Box-side served-model quality probe — runs ON the remote GPU host (invoked over SSH by
``engine.quality_remote``), NOT on the local no-GPU driver. There, ``127.0.0.1`` IS the GPU box, so
serving + querying the served model is correct (mirroring how ``run_remote_bench`` runs the native
bench on the box).

``probe(flat_config, prompts)`` serves the model with the candidate's full quality-affecting engine
config (vanilla scheduler — scheduling doesn't change the emitted tokens) and queries the FROZEN
calibration battery, returning ``{perplexity, task_em, continuations}`` (the driver computes the
candidate-vs-baseline output agreement from the per-config continuations). ``main()`` reads
``{config, prompts}`` JSON on stdin and prints the result JSON (``null`` on any failure / no GPU),
which the driver treats as NOT MEASURED — the accept gate then hard-fails, never a false adopt.
"""
from __future__ import annotations

import json
import math
import sys

from vllm_evolve.bench.config import VANILLA, build_bench_config
from vllm_evolve.bench.native import SERVED_NAME, serve_vllm
from vllm_evolve.engine.quality_runner import calibration_set

_MAX_TOKENS = 8
_PORT = 8270

# build_bench_config takes these as NAMED kwargs, so they must not also flow through **overrides.
_BC_NAMED = {"model", "gpus", "profile", "policy", "policy_path", "scheduler_cls",
             "runner_kind", "remote", "port", "primary_metric"}


def _http_post(url: str, payload: dict):
    """One POST to the served model. Wrapped so tests can stub the HTTP boundary."""
    import httpx
    r = httpx.post(url, json=payload, timeout=120.0)
    r.raise_for_status()
    return r.json()


def _expected_for(prompt: str) -> str:
    for c in calibration_set():
        if c.get("prompt") == prompt:
            return c.get("expected", "") or ""
    return ""


def _query(base_url: str, prompt: str) -> dict | None:
    """One greedy ``/v1/completions`` call with echo + logprobs. Returns
    ``{"logprobs": [...], "continuation": str}`` or ``None`` on any failure / missing field."""
    try:
        body = _http_post(f"{base_url}/v1/completions",
                          {"model": SERVED_NAME, "prompt": prompt, "max_tokens": _MAX_TOKENS,
                           "temperature": 0.0, "logprobs": 1, "echo": True})
        ch = body["choices"][0]
    except Exception:        # noqa: BLE001 - honest measurement failure -> NOT MEASURED
        return None
    lps = (ch.get("logprobs") or {}).get("token_logprobs")
    text = ch.get("text")
    if not lps or text is None:
        return None
    valid = [lp for lp in lps if isinstance(lp, (int, float))]
    if not valid:
        return None
    continuation = text[len(prompt):] if text.startswith(prompt) else text
    return {"logprobs": valid, "continuation": continuation}


def _perplexity(all_logprobs: list[float]) -> float:
    return math.exp(-sum(all_logprobs) / len(all_logprobs))


def _task_em(prompts: list[str], conts: list[str]) -> float:
    if not prompts:
        return 0.0
    hits = 0
    for p, c in zip(prompts, conts):
        exp = _expected_for(p).strip().lower()
        if exp and exp in (c or "").strip().lower():
            hits += 1
    return hits / len(prompts)


def _extra_serve_args(e) -> list[str]:
    """Quality-affecting engine levers that ``serve_vllm`` does NOT take as a structured param —
    rendered into ``extra_args`` so a ``config:quantization`` / ``config:kv_cache_dtype`` (etc.)
    candidate is SERVED with its real representation when its quality is certified. Mirrors
    ``BenchConfig.to_serve_args`` (guarded by the candidate-config-reaches-serve test)."""
    extra: list[str] = []
    if e.max_num_batched_tokens:
        extra += ["--max-num-batched-tokens", str(e.max_num_batched_tokens)]
    if e.quantization:
        extra += ["--quantization", e.quantization]
    if e.kv_cache_dtype:
        extra += ["--kv-cache-dtype", e.kv_cache_dtype]
    if e.enable_prefix_caching is True:
        extra += ["--enable-prefix-caching"]
    elif e.enable_prefix_caching is False:
        extra += ["--no-enable-prefix-caching"]
    if e.enable_chunked_prefill is True:
        extra += ["--enable-chunked-prefill"]
    elif e.enable_chunked_prefill is False:
        extra += ["--no-enable-chunked-prefill"]
    if e.async_scheduling is True:
        extra += ["--async-scheduling"]
    elif e.async_scheduling is False:
        extra += ["--no-async-scheduling"]
    return extra


def bench_config_for(flat: dict):
    """A VANILLA BenchConfig (no scheduler plugin) carrying the candidate's full quality-affecting
    engine config, built from the SAME flat contract bench/native consumes — so artifact model
    selection + every executable engine lever match the config being certified."""
    overrides = {k: v for k, v in flat.items() if k not in _BC_NAMED}
    config = build_bench_config(
        runner_kind=VANILLA,
        scheduler_cls=None,
        model=flat["model"],
        gpus=str(flat.get("gpus", "0")),
        profile=str(flat.get("profile", "throughput")),
        remote=flat.get("remote"),
        **overrides,
    )
    # A frozen formal suite may use non-default Jiusi paths. Preserve the
    # execution environment as well as the engine knobs; otherwise quality can
    # accidentally run against a different checkout/conda/cache environment.
    for field in (
        "remote_repo",
        "conda_env",
        "conda_sh",
        "remote_workspace",
        "local_artifact_root",
        "hf_endpoint",
    ):
        if flat.get(field) is not None:
            setattr(config.runner, field, flat[field])
    return config


def probe(flat_config: dict, prompts: list[str]) -> dict | None:
    """Serve the candidate's representation on THIS host and measure the calibration battery.
    Returns ``{perplexity, task_em, continuations}`` or ``None`` (NOT MEASURED) on any failure."""
    flat = dict(flat_config or {})
    if not flat.get("model"):
        return None
    bc = bench_config_for(flat)
    e = bc.engine
    try:
        with serve_vllm(bc.model_to_serve(), "", gpus=bc.environment.gpus, port=_PORT,
                        scheduler_cls=None, gpu_memory_utilization=e.gpu_memory_utilization,
                        max_model_len=e.max_model_len, max_num_seqs=e.max_num_seqs,
                        tensor_parallel_size=e.tensor_parallel_size, enforce_eager=e.enforce_eager,
                        extra_args=_extra_serve_args(e)) as handle:
            base_url = f"http://127.0.0.1:{handle.port}"
            rows = [_query(base_url, p) for p in prompts]
    except Exception:        # noqa: BLE001 - serving failed (no GPU here) -> NOT MEASURED
        return None
    if not rows or any(r is None for r in rows):
        return None
    all_lps = [lp for r in rows for lp in r["logprobs"]]
    if not all_lps:
        return None
    conts = [r["continuation"] for r in rows]
    return {"perplexity": _perplexity(all_lps), "task_em": _task_em(prompts, conts),
            "continuations": conts}


def main(argv: list[str] | None = None) -> int:
    """Read ``{config, prompts}`` JSON on stdin; print probe result JSON (``null`` on failure)."""
    try:
        payload = json.loads(sys.stdin.read())
        result = probe(payload.get("config", {}), payload.get("prompts", []))
    except Exception:        # noqa: BLE001 - any failure is honestly NOT MEASURED
        result = None
    print(json.dumps(result))
    return 0


if __name__ == "__main__":   # pragma: no cover - box-side entry point
    raise SystemExit(main())
