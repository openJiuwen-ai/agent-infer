"""Round 6: collect_profile routes every wired lever through BenchConfig (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench.config import WIRED_LEVERS  # noqa: E402


def test_collect_profile_routes_levers_through_benchconfig(monkeypatch):
    # Codex R5: the profile path must build a BenchConfig and route through
    # run_remote_bench_config — no loose-dict drop of newly allowed levers.
    import vllm_evolve.bench.dispatch as d
    from vllm_evolve.bench.dispatch import RemoteBench

    captured = {}

    def fake_rrbc(config):
        captured["bc"] = config
        return RemoteBench({"raw_per_seed_metrics": [], "output_throughput_tok_s": 100.0,
                            "bench_config": config.to_dict()}, "log", "/e", "/l", remote_cmd="cmd")

    monkeypatch.setattr(d, "run_remote_bench_config", fake_rrbc)
    from vllm_evolve.engine.profile import collect_profile

    cfg = {"model": "m7b", "max_num_seqs": 256, "gpu_memory_utilization": 0.92,
           "quantization": "fp8", "kv_cache_dtype": "fp8", "max_num_batched_tokens": 4096,
           "enable_chunked_prefill": True, "enable_prefix_caching": True, "enforce_eager": True,
           "tensor_parallel_size": 2, "max_model_len": 8192, "concurrency": 64, "n_requests": 100}
    prof = collect_profile(cfg)

    bc = captured["bc"]
    # every engine lever the search set is on the BenchConfig (not dropped)
    assert bc.engine.quantization == "fp8" and bc.engine.kv_cache_dtype == "fp8"
    assert bc.engine.max_num_seqs == 256 and bc.engine.max_num_batched_tokens == 4096
    assert bc.engine.enable_chunked_prefill is True and bc.engine.enforce_eager is True
    assert bc.engine.gpu_memory_utilization == 0.92 and bc.engine.max_model_len == 8192
    # workload levers too
    assert bc.workload.concurrency == 64 and bc.workload.n_requests == 100
    # the rendered command carries EVERY WIRED engine lever (the proven boundary)
    from vllm_evolve.bench.dispatch import _config_extra_serve_args
    rendered = " ".join(bc.to_serve_args()) + " " + " ".join(_config_extra_serve_args(bc))
    for lever in WIRED_LEVERS:
        flag = "--" + lever.replace("_", "-")
        assert flag in rendered, f"{lever} dropped before the command"
    # the Profile carries bench_config provenance for the same_caliber A/B
    assert prof.bench_config is not None
