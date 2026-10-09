"""AC7 (box-gated, NON-gating for COMPLETE): real-vLLM end-to-end integration.

This is the ONE test that exercises the real serve+bench path against actual vLLM. It is GPU-gated:
off-hardware it is SKIPPED (never a fabricated pass); on a GPU host (``VLLM_EVOLVE_GPU=1``) it
renders the seed scheduling plugin, serves a tiny model on the real path, drives a small bench, and
asserts the nonce'd plugin marker appears in the REAL serve log plus a ``source='real_vllm'`` /
``outcome_class='eval_result'`` artifact with non-empty metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))


@pytest.mark.requires_gpu
def test_real_vllm_end_to_end():  # pragma: no cover - runs only on a GPU host
    from vllm_evolve.bench.native import PLUGIN_INVOKED_MARKER, native_bench
    from vllm_evolve.bench.profiles import load_profile
    seed_source = (REPO_ROOT / "targets" / "scheduling" / "seed.py").read_text(encoding="utf-8")
    # native_bench takes a loaded BenchProfile (the CLI passes load_profile(...), not a name string)
    profile = load_profile(str(REPO_ROOT / "config" / "bench" / "profiles" / "throughput.yaml"))
    er, log = native_bench(
        seed_source, profile, runner_kind="candidate",
        model="facebook/opt-125m", n_requests=8, max_seeds=1, port=8231)
    # the plugin actually loaded + ran inside real vLLM
    assert PLUGIN_INVOKED_MARKER in log
    # and produced a real, clean eval_result with real numbers
    assert er["source"] == "real_vllm"
    assert er["outcome_class"] == "eval_result"
    assert er["raw_per_seed_metrics"]
    assert er["aggregate_metrics"].get("median") is not None
