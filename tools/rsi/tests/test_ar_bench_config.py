"""Round 1: BenchConfig as execution contract + ve bench wiring (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
for p in (str(SRC), str(REPO_ROOT)):
    if p not in sys.path:  # pragma: no cover - import path bootstrap
        sys.path.insert(0, p)

import pytest  # noqa: E402

from vllm_evolve.bench.config import (  # noqa: E402
    CANDIDATE,
    STRONG_BASELINE,
    VANILLA,
    WIRED_LEVERS,  # noqa: E402
    build_bench_config,
)
from vllm_evolve.bench.dispatch import (  # noqa: E402
    _compile_cache_namespace,
    _config_extra_serve_args,
    run_remote_bench_config,
)

# WIRED engine lever -> the serve flag that must appear in the executed command.
_LEVER_FLAG = {
    "quantization": "--quantization", "kv_cache_dtype": "--kv-cache-dtype",
    "max_num_batched_tokens": "--max-num-batched-tokens", "max_num_seqs": "--max-num-seqs",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "enable_prefix_caching": "--enable-prefix-caching",
    "enable_chunked_prefill": "--enable-chunked-prefill",
    "tensor_parallel_size": "--tensor-parallel-size", "max_model_len": "--max-model-len",
    "enforce_eager": "--enforce-eager",
}


def test_every_wired_lever_and_workload_reaches_the_command(monkeypatch, tmp_path):
    # Codex R2: no WIRED lever may be silently dropped before dispatch. Build a config
    # with every lever + workload knob set, capture the remote command, assert each flag.
    import vllm_evolve.bench.dispatch as d
    captured = {}
    monkeypatch.setattr(d, "_scp", lambda *a, **k: None)
    monkeypatch.setattr(d, "_ensure_remote_workspace", lambda *a, **k: None)

    def fake_run(cmd, timeout_s):
        captured["cmd"] = cmd[-1]
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(d, "_run", fake_run)
    pol = tmp_path / "p.py"
    pol.write_text("def schedule_batch(*a, **k):\n    return None\n", encoding="utf-8")

    cfg = build_bench_config(
        runner_kind=CANDIDATE, policy_path=str(pol),
        scheduler_cls="generated_scheduler.EvolvedScheduler",
        quantization="fp8", kv_cache_dtype="fp8", max_num_batched_tokens=4096,
        max_num_seqs=128, gpu_memory_utilization=0.92, enable_chunked_prefill=True,
        enable_prefix_caching=True, tensor_parallel_size=2, max_model_len=8192,
        enforce_eager=True, async_scheduling=True, concurrency=64,
        n_requests=100, gpus="0,1")
    with pytest.raises(RuntimeError, match="stop-after-capture"):
        run_remote_bench_config(cfg)
    cmd = captured["cmd"]
    # every WIRED engine lever has a mapped flag and it appears in the command
    assert set(_LEVER_FLAG) == set(WIRED_LEVERS)
    for lever, flag in _LEVER_FLAG.items():
        assert flag in cmd, f"{lever} ({flag}) dropped before dispatch"
    # workload knobs reach the command too
    assert "--concurrency" in cmd and "64" in cmd
    assert "--n-requests" in cmd and "100" in cmd


def test_run_remote_bench_config_augments_eval_result_provenance(monkeypatch):
    # Codex R3: real eval_results must carry BenchConfig provenance so same_caliber is
    # enforceable end-to-end. Mock the SSH round-trip; assert the augmentation.
    import vllm_evolve.bench.dispatch as d
    from vllm_evolve.bench.dispatch import RemoteBench

    def fake_rrb(*a, **k):
        return RemoteBench({"primary_metric": "tok_s", "raw_per_seed_metrics": []},
                           "log", "/e.json", "/l.log", remote_cmd="ssh box native --runner-kind")

    monkeypatch.setattr(d, "run_remote_bench", fake_rrb)
    rb = run_remote_bench_config(build_bench_config(
        runner_kind=STRONG_BASELINE, model="m7b", quantization="fp8"))
    er = rb.eval_result
    assert "bench_config" in er and er["runner_kind"] == "strong_baseline"
    assert er["model_served"] == "m7b" and "--quantization" in er["rendered_serve_args"]
    assert er["remote_cmd"] == "ssh box native --runner-kind"


def test_strong_baseline_serves_artifact_and_renders_levers():
    c = build_bench_config(runner_kind=STRONG_BASELINE, quantization="fp8",
                           kv_cache_dtype="fp8", model_artifact_kind="fp8-ckpt",
                           quantized_model_id="org/m-fp8", max_num_batched_tokens=4096)
    args = c.to_serve_args()
    assert "--quantization" in args and "--kv-cache-dtype" in args
    assert "--max-num-batched-tokens" in args
    assert c.model_to_serve() == "org/m-fp8"        # serves the quantized artifact
    assert c.is_baseline() and "--scheduler-cls" not in args   # baseline: no plugin


def test_candidate_attaches_plugin_and_serves_base_model():
    c = build_bench_config(runner_kind=CANDIDATE,
                           scheduler_cls="generated_scheduler.EvolvedScheduler",
                           model="facebook/opt-125m")
    assert c.model_to_serve() == "facebook/opt-125m"
    assert "--scheduler-cls" in c.to_serve_args()
    assert not c.is_baseline()


def test_extra_serve_args_render_all_levers():
    c = build_bench_config(quantization="fp8", kv_cache_dtype="fp8",
                           enable_chunked_prefill=True, max_num_batched_tokens=8192)
    extra = _config_extra_serve_args(c)
    assert "--quantization" in extra and "--kv-cache-dtype" in extra
    assert "--enable-chunked-prefill" in extra and "--max-num-batched-tokens" in extra


def test_compile_cache_is_isolated_by_physical_gpu_but_not_policy():
    common = {
        "model": "/models/qwen32b",
        "tensor_parallel_size": 1,
        "max_num_seqs": 150,
        "gpu_memory_utilization": 0.86,
        "max_model_len": 4096,
        "extra_serve_args": ["--enforce-eager"],
    }
    gpu0 = _compile_cache_namespace(gpus="0", **common)
    gpu1 = _compile_cache_namespace(gpus="1", **common)
    gpu0_again = _compile_cache_namespace(gpus="0", **common)

    assert gpu0 != gpu1
    assert gpu0 == gpu0_again


def test_replica_member_barrier_reaches_native_command(monkeypatch):
    import vllm_evolve.bench.dispatch as d

    captured = {}
    monkeypatch.setattr(d, "_scp", lambda *a, **k: None)
    monkeypatch.setattr(d, "_ensure_remote_workspace", lambda *a, **k: None)

    def fake_run(cmd, timeout_s):
        captured["cmd"] = cmd[-1]
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(d, "_run", fake_run)
    cfg = build_bench_config(runner_kind=STRONG_BASELINE, gpus="0")
    cfg.workload.workload_spec = {
        "parallel_mode": "replica_member",
        "replica_member": {
            "barrier_id": "barrier123",
            "index": 0,
            "count": 2,
            "barrier_timeout_s": 300,
        },
    }

    with pytest.raises(RuntimeError, match="stop-after-capture"):
        run_remote_bench_config(cfg)

    assert "--replica-barrier-id barrier123" in captured["cmd"]
    assert "--replica-index 0" in captured["cmd"]
    assert "--replica-count 2" in captured["cmd"]


def test_dual_replica_config_routes_to_group_runner(monkeypatch):
    import vllm_evolve.bench.replica_group as group
    from vllm_evolve.bench.dispatch import RemoteBench

    sentinel = RemoteBench({}, "", "", "")
    captured = {}

    def fake_group(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(group, "run_remote_replica_group", fake_group)
    cfg = build_bench_config(
        runner_kind=STRONG_BASELINE,
        gpus="0,1",
        tensor_parallel_size=1,
    )
    cfg.workload.workload_spec = {"parallel_mode": "dual_replica"}

    assert run_remote_bench_config(cfg, git_sha="frozen") is sentinel
    assert captured["config"] is cfg
    assert captured["git_sha"] == "frozen"


def test_dual_replica_auto_selection_is_bound_before_group_dispatch(monkeypatch):
    import vllm_evolve.bench.dispatch as d
    import vllm_evolve.bench.replica_group as group
    from vllm_evolve.bench.dispatch import RemoteBench

    sentinel = RemoteBench({}, "", "", "")
    captured = {}

    def fake_bind(config):
        config.environment.gpus = "4,5"
        config.environment.gpu_selection_mode = "auto"
        config.environment.gpu_selection_evidence = {
            "mode": "auto",
            "resolved": "4,5",
        }

    def fake_group(config, **kwargs):
        captured["gpus"] = config.environment.gpus
        return sentinel

    monkeypatch.setattr(d, "bind_auto_gpu_selection", fake_bind)
    monkeypatch.setattr(group, "run_remote_replica_group", fake_group)
    cfg = build_bench_config(
        runner_kind=STRONG_BASELINE,
        gpus="auto",
        tensor_parallel_size=1,
    )
    cfg.workload.workload_spec = {"parallel_mode": "dual_replica"}

    result = run_remote_bench_config(cfg, git_sha="frozen")

    assert captured["gpus"] == "4,5"
    assert cfg.environment.gpus == "4,5"
    assert result.eval_result["gpu_selection"]["resolved"] == "4,5"


def test_all_runner_kinds_build_an_attempted_command(monkeypatch, tmp_path):
    # Codex R1: baselines must REACH the native/SSH boundary (not a pre-empting guard).
    # Monkeypatch the boundary and prove all three kinds build an attempted command;
    # baselines omit --policy, candidate includes it.
    import vllm_evolve.bench.dispatch as d

    captured = {}
    monkeypatch.setattr(d, "_scp", lambda *a, **k: None)
    monkeypatch.setattr(d, "_ensure_remote_workspace", lambda *a, **k: None)

    def fake_run(cmd, timeout_s):
        captured["cmd"] = cmd[-1]   # the remote command string (last ssh arg)
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(d, "_run", fake_run)
    pol = tmp_path / "p.py"
    pol.write_text("def schedule_batch(*a, **k):\n    return None\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stop-after-capture"):
        run_remote_bench_config(build_bench_config(
            runner_kind=CANDIDATE, policy_path=str(pol),
            scheduler_cls="generated_scheduler.EvolvedScheduler"))
    assert "--runner-kind candidate" in captured["cmd"] and "--policy" in captured["cmd"]

    for kind in (VANILLA, STRONG_BASELINE):
        with pytest.raises(RuntimeError, match="stop-after-capture"):
            run_remote_bench_config(build_bench_config(runner_kind=kind))
        assert f"--runner-kind {kind}" in captured["cmd"]
        assert "--policy" not in captured["cmd"]   # baseline: no plugin/policy


def test_run_remote_bench_config_forwards_remote_environment(monkeypatch):
    # Codex review P2: the SSH run must use the full recorded remote environment
    # (recorded in provenance), NOT the process defaults — else results come from the wrong env.
    import vllm_evolve.bench.dispatch as d
    from vllm_evolve.bench.dispatch import RemoteBench
    captured = {}

    def fake_rrb(*a, **k):
        captured.update(k)
        return RemoteBench({"raw_per_seed_metrics": []}, "log", "/e.json", "/l.log")

    monkeypatch.setattr(d, "run_remote_bench", fake_rrb)
    cfg = build_bench_config(runner_kind=STRONG_BASELINE, model="m")
    cfg.runner.remote_repo = "/other/repo"
    cfg.runner.conda_env = "other-env"
    cfg.runner.conda_sh = "/other/conda.sh"
    cfg.runner.remote_workspace = "/other/workspace"
    cfg.runner.local_artifact_root = "/other/artifacts"
    cfg.runner.hf_endpoint = "https://hf.example.test"
    cfg.statistical.timeout_s = 4321
    run_remote_bench_config(cfg, git_sha="frozen-suite-sha")
    assert captured["remote_repo"] == "/other/repo"
    assert captured["conda_env"] == "other-env"
    assert captured["conda_sh"] == "/other/conda.sh"
    assert captured["remote_workspace"] == "/other/workspace"
    assert captured["local_artifact_root"] == "/other/artifacts"
    assert captured["hf_endpoint"] == "https://hf.example.test"
    assert captured["timeout_s"] == 4321
    assert captured["primary_metric"] == cfg.statistical.primary_metric
    assert captured["slo"] == cfg.statistical.slo
    assert captured["git_sha"] == "frozen-suite-sha"


def test_runner_config_honors_ve_env_overrides(monkeypatch):
    # Codex review P2 follow-up: a default-built BenchConfig (CLI/profile/autopt path) must record
    # the user's VE_* remote env in RunnerConfig + provenance, so the forwarded SSH run both MATCHES
    # provenance AND honors the documented VE_REMOTE_REPO / VE_CONDA_ENV / VE_HF_ENDPOINT overrides
    # (instead of the hard-coded defaults).
    monkeypatch.setenv("VE_REMOTE_REPO", "/custom/repo")
    monkeypatch.setenv("VE_CONDA_ENV", "custom-env")
    monkeypatch.setenv("VE_CONDA_SH", "/custom/conda.sh")
    monkeypatch.setenv("VE_REMOTE_WORKSPACE", "/custom/workspace")
    monkeypatch.setenv("VE_LOCAL_ARTIFACT_ROOT", "/custom/artifacts")
    monkeypatch.setenv("VE_HF_ENDPOINT", "https://custom.hf")
    bc = build_bench_config(runner_kind=STRONG_BASELINE, model="m")
    assert bc.runner.remote_repo == "/custom/repo"
    assert bc.runner.conda_env == "custom-env"
    assert bc.runner.conda_sh == "/custom/conda.sh"
    assert bc.runner.remote_workspace == "/custom/workspace"
    assert bc.runner.local_artifact_root == "/custom/artifacts"
    assert bc.runner.hf_endpoint == "https://custom.hf"
    prov = bc.provenance()["bench_config"]["runner"]
    assert prov["remote_repo"] == "/custom/repo"   # provenance == the env the SSH run executes in


def test_build_bench_config_honors_ve_remote(monkeypatch):
    # Codex review P2: default ve bench/calibrate (no --remote) must SSH to VE_REMOTE, and
    # provenance must record that host (not the literal gpu-host).
    monkeypatch.setenv("VE_REMOTE", "my-box")
    bc = build_bench_config(runner_kind=STRONG_BASELINE, model="m")     # no explicit remote
    assert bc.runner.remote == "my-box"
    assert bc.provenance()["bench_config"]["runner"]["remote"] == "my-box"
    # an explicit remote still wins over the env
    bc2 = build_bench_config(runner_kind=STRONG_BASELINE, model="m", remote="explicit-box")
    assert bc2.runner.remote == "explicit-box"


def test_enforce_eager_lever_is_honored():
    # Codex review P2: enforce_eager=False must NOT render --enforce-eager (vLLM default cudagraph),
    # True must, and serve_vllm must not independently force eager (default False) — else False runs
    # eager anyway.
    import inspect

    from vllm_evolve.bench.dispatch import _config_extra_serve_args
    from vllm_evolve.bench.native import serve_vllm
    assert inspect.signature(serve_vllm).parameters["enforce_eager"].default is False
    on = build_bench_config(runner_kind=STRONG_BASELINE, enforce_eager=True)
    off = build_bench_config(runner_kind=STRONG_BASELINE, enforce_eager=False)
    assert "--enforce-eager" in on.to_serve_args()
    assert "--enforce-eager" not in off.to_serve_args()
    assert "--enforce-eager" in _config_extra_serve_args(on)         # the wired remote command too
    assert "--enforce-eager" not in _config_extra_serve_args(off)


def test_run_remote_bench_config_persists_bench_config_to_file(monkeypatch, tmp_path):
    # Codex review P1: the FILE at local_eval_path must carry the augmented bench_config — ve bench
    # emits that path and ve compare reads the FILE (not the in-memory result), else it would fail
    # same_caliber_unverifiable on a real run.
    import json

    import vllm_evolve.bench.dispatch as d
    from vllm_evolve.bench.dispatch import RemoteBench
    raw = tmp_path / "e.json"
    raw.write_text(json.dumps({"raw_per_seed_metrics": []}), encoding="utf-8")  # raw remote JSON

    def fake_rrb(*a, **k):
        return RemoteBench(json.loads(raw.read_text(encoding="utf-8")), "log", str(raw),
                           str(tmp_path / "l.log"), remote_cmd="ssh box")

    monkeypatch.setattr(d, "run_remote_bench", fake_rrb)
    run_remote_bench_config(build_bench_config(runner_kind=STRONG_BASELINE, model="m"))
    on_disk = json.loads(raw.read_text(encoding="utf-8"))
    assert "bench_config" in on_disk and on_disk["runner_kind"] == "strong_baseline"
    assert on_disk["remote_cmd"] == "ssh box"


def test_max_seeds_threads_to_run_remote_bench(monkeypatch):
    # Codex review P2: --max-seeds must reach run_remote_bench (budget cap), not be filtered away.
    import vllm_evolve.bench.dispatch as d
    from vllm_evolve.bench.dispatch import RemoteBench
    cfg = build_bench_config(runner_kind=STRONG_BASELINE, model="m", max_seeds=3)
    assert cfg.statistical.max_seeds == 3
    captured = {}

    def fake_rrb(*a, **k):
        captured.update(k)
        return RemoteBench({"raw_per_seed_metrics": []}, "log", "", "")

    monkeypatch.setattr(d, "run_remote_bench", fake_rrb)
    run_remote_bench_config(cfg)
    assert captured["max_seeds"] == 3


def test_materialized_workload_threads_to_remote_bench(monkeypatch, tmp_path):
    import vllm_evolve.bench.dispatch as d
    from vllm_evolve.bench.dispatch import RemoteBench

    workload = tmp_path / "workload.json"
    workload.write_text('{"requests":[]}\n', encoding="utf-8")
    cfg = build_bench_config(
        runner_kind=STRONG_BASELINE,
        model="m",
        trace_path=str(workload),
        load_mode="open",
    )
    captured = {}

    def fake_rrb(*args, **kwargs):
        captured.update(kwargs)
        return RemoteBench({"raw_per_seed_metrics": []}, "log", "", "")

    monkeypatch.setattr(d, "run_remote_bench", fake_rrb)
    run_remote_bench_config(cfg)
    assert captured["workload_path"] == str(workload)
    assert captured["bench_config"]["workload"]["load_mode"] == "open"


def test_chunked_prefill_lever_honors_false():
    # Codex review P2: enable_chunked_prefill=False must render the DISABLE flag (not nothing) so
    # explicit disable isn't lost on vLLM builds where chunked prefill defaults on; None leaves it.
    from vllm_evolve.bench.dispatch import _config_extra_serve_args
    on = build_bench_config(runner_kind=STRONG_BASELINE, enable_chunked_prefill=True)
    off = build_bench_config(runner_kind=STRONG_BASELINE, enable_chunked_prefill=False)
    default = build_bench_config(runner_kind=STRONG_BASELINE)        # None -> leave vLLM default
    assert "--enable-chunked-prefill" in on.to_serve_args()
    assert "--no-enable-chunked-prefill" in off.to_serve_args()
    assert "chunked-prefill" not in " ".join(default.to_serve_args())
    assert "--no-enable-chunked-prefill" in _config_extra_serve_args(off)
    assert "--enable-chunked-prefill" in _config_extra_serve_args(on)


def test_eval_artifact_path_is_unique_per_config(monkeypatch, tmp_path):
    # Codex review P2: the SAME candidate policy under different profiles must NOT collide on one
    # one /tmp eval path (else a later run overwrites the path an earlier ve bench emitted).
    import re

    import vllm_evolve.bench.dispatch as d
    from vllm_evolve.bench.dispatch import run_remote_bench
    monkeypatch.setattr(d, "_scp", lambda *a, **k: None)
    monkeypatch.setattr(d, "_ensure_remote_workspace", lambda *a, **k: None)
    captured = []

    def fake_run(cmd, timeout_s):
        captured.append(cmd[-1])
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(d, "_run", fake_run)
    pol = tmp_path / "p.py"
    pol.write_text("def schedule_batch(rs, st):\n    return None\n", encoding="utf-8")
    for prof in ("throughput", "latency"):
        with pytest.raises(RuntimeError, match="stop-after-capture"):
            run_remote_bench(str(pol), prof, runner_kind="candidate")
    outs = [re.search(r"--out (\S+)", c).group(1) for c in captured]
    assert outs[0] != outs[1], outs        # different profile -> different artifact path
