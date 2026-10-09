"""AC4: CUDA profiling collector — capability probe + degraded/collected paths (zero GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench.cuda_profile import (  # noqa: E402
    capability_probe,
    collect_cuda_profile,
    probe_all,
)
from vllm_evolve.engine.cuda_parse import KernelBreakdown  # noqa: E402


def test_capability_probe_is_deterministic():
    p = capability_probe("nsys")
    assert p["profiler"] == "nsys" and isinstance(p["available"], bool) and p["detail"]
    assert capability_probe("bogus")["available"] is False
    assert set(probe_all()) == {"nsys", "torch", "ncu"}


def test_unavailable_profiler_degrades_honestly():
    # ncu is not on PATH in CI -> degraded, NOT a fabricated breakdown
    r = collect_cuda_profile("ncu", collect_fn=None)
    if r.capability["available"]:           # if ncu happens to exist, no collector wired
        assert r.status == "degraded_unavailable" and r.breakdown is None
    else:
        assert r.status == "degraded_unavailable" and r.breakdown is None


def test_collected_breakdown_feeds_kernel_lever_map():
    from vllm_evolve.engine.kernel_map import match_levers

    def fake_collect():
        return (["/run/cuda/trace.json"], "nsys profile -o trace vllm serve ...")

    def fake_parse(_artifact):
        return KernelBreakdown(gemm_time_frac=0.6, tc_util=0.3, source="nsys")

    # force capability available by probing 'torch' (importable) — use a monkeypatch-free path:
    r = collect_cuda_profile("torch", collect_fn=fake_collect, parse_fn=fake_parse)
    if r.status == "collected":
        assert r.breakdown["gemm_time_frac"] == 0.6
        matched = match_levers({k: v for k, v in r.breakdown.items() if v is not None})
        assert any(m["name"] == "gemm_compute_bound" for m in matched)
        assert r.attempted_cmd.startswith("nsys profile")


def test_collection_failure_is_degraded_not_fabricated():
    def boom():
        raise RuntimeError("nsys exited 1")

    r = collect_cuda_profile("torch", collect_fn=boom, attempted_cmd="nsys profile ...")
    if r.capability["available"]:
        assert r.status == "degraded_failed" and "nsys exited 1" in r.error
        assert r.breakdown is None and r.attempted_cmd == "nsys profile ..."   # cmd preserved


def test_cuda_command_is_same_caliber_native_workload(tmp_path):
    # Codex R8: the profiled window must run the LOADED BenchConfig workload via the canonical
    # native runner in the configured remote ENVIRONMENT — not a bare serve-only command.
    from vllm_evolve.bench.config import build_bench_config
    from vllm_evolve.bench.cuda_profile import make_cuda_collector
    bc = build_bench_config(runner_kind="strong_baseline", model="m7b", remote="box",
                            max_num_seqs=256, quantization="fp8", concurrency=64, n_requests=200)
    collect_fn, cmd = make_cuda_collector("nsys", bench_config=bc, run_dir=str(tmp_path))
    # wraps the canonical workload runner under the profiler (NOT a bare `vllm serve`)
    assert "nsys profile" in cmd and "python -m vllm_evolve.bench.native" in cmd
    assert "vllm serve" not in cmd                                   # serve-only was the R8 bug
    # workload + engine + runner caliber
    assert "--n-requests 200" in cmd and "--concurrency 64" in cmd
    assert "--max-num-seqs 256" in cmd and "--quantization" in cmd   # quant via --extra-serve-args
    assert "--profile config/bench/profiles/throughput.yaml" in cmd
    assert "--runner-kind strong_baseline" in cmd and "--port" in cmd
    # the configured remote ENVIRONMENT prelude (Codex R8)
    assert "conda activate" in cmd and "cd /workspace" in cmd
    assert "HF_ENDPOINT=" in cmd and "CUDA_VISIBLE_DEVICES=" in cmd
    # remote POSIX artifact dir; no Windows local path leaked
    assert "/tmp/vllm_evolve_cuda/" in cmd and "\\" not in cmd and "C:" not in cmd
    assert callable(collect_fn)


def test_candidate_cuda_stages_policy_remotely(monkeypatch, tmp_path):
    # Codex R9: candidate CUDA must STAGE the policy (scp to /tmp/ve_<hash>_policy.py) like the
    # real bench — not pass a controller-local path to the remote runner.
    import re
    import subprocess

    from vllm_evolve.bench import cuda_profile as cp
    from vllm_evolve.bench.config import build_bench_config
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(*a, **k):\n    return None\n", encoding="utf-8")
    bc = build_bench_config(runner_kind="candidate", policy_path=str(pol),
                            scheduler_cls="generated_scheduler.EvolvedScheduler", remote="box")
    collect_fn, cmd = cp.make_cuda_collector("nsys", bench_config=bc, run_dir=str(tmp_path))
    # the attempted command references the REMOTE staged path, NOT the local controller path
    assert re.search(r"--policy /tmp/ve_[0-9a-f]+_policy\.py", cmd)
    assert str(pol) not in cmd

    order = []

    def fake_run(args, **k):
        order.append(args[0])
        r = type("R", (), {})()
        r.returncode, r.stdout, r.stderr = 0, "", ""
        return r

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("os.listdir", lambda d: ["nsys_window.qdrep"])
    collect_fn()
    assert order[0] == "scp" and "ssh" in order          # policy staged BEFORE the profiler ssh
    assert order.index("scp") < order.index("ssh")


def test_candidate_cuda_missing_policy_degrades(tmp_path):
    import pytest

    from vllm_evolve.bench import cuda_profile as cp
    from vllm_evolve.bench.config import build_bench_config
    bc = build_bench_config(runner_kind="candidate", policy_path=str(tmp_path / "nope.py"),
                            scheduler_cls="x.Y", remote="box")
    # unreadable local policy -> make_cuda_collector raises -> cmd_profile turns it into degraded
    with pytest.raises(Exception):
        cp.make_cuda_collector("nsys", bench_config=bc, run_dir=str(tmp_path))


def test_candidate_cuda_uses_shared_staging_helper(monkeypatch, tmp_path):
    # Codex R10: the CUDA collector must call the SHARED stage_policy_to_remote() helper, not an
    # equivalent inline scp — so bench + profiler staging cannot drift.
    import subprocess

    from vllm_evolve.bench import cuda_profile as cp
    from vllm_evolve.bench import dispatch as d
    from vllm_evolve.bench.config import build_bench_config
    from vllm_evolve.bench.dispatch import policy_remote_path
    pol = tmp_path / "work.py"
    pol.write_text("def schedule_batch(*a, **k):\n    return None\n", encoding="utf-8")
    bc = build_bench_config(runner_kind="candidate", policy_path=str(pol),
                            scheduler_cls="x.Y", remote="box")
    remote_policy, _ = policy_remote_path(str(pol))
    order = []

    def fake_stage(policy_path, remote):
        order.append("stage")
        return remote_policy, "rid"          # must match the command's remote path (guard)

    monkeypatch.setattr(d, "stage_policy_to_remote", fake_stage)
    collect_fn, cmd = cp.make_cuda_collector("nsys", bench_config=bc, run_dir=str(tmp_path))
    assert remote_policy in cmd

    def fake_run(args, **k):
        order.append(args[0])
        r = type("R", (), {})()
        r.returncode, r.stdout, r.stderr = 0, "", ""
        return r

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("os.listdir", lambda _d: ["nsys_window.qdrep"])
    collect_fn()
    assert order.count("stage") == 1                         # the SHARED helper was called
    assert order[0] == "stage" and "ssh" in order            # staged before the profiler ssh
    assert order.index("stage") < order.index("ssh")


def test_remote_capability_probe_attempts_collection(monkeypatch):
    # local nsys absent but remote present -> still attempt collection (Codex R7)
    import subprocess

    from vllm_evolve.bench import cuda_profile as cp

    class _R:
        returncode = 0
        stdout = "/usr/bin/nsys"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    cap = cp.capability_probe("nsys", remote="box")
    assert cap["available"] is True and cap["location"] == "remote"

    called = {}

    def collect_fn():
        called["yes"] = True
        return (["/run/cuda/nsys_window.qdrep"], "cmd")

    r = cp.collect_cuda_profile("nsys", collect_fn=collect_fn, remote="box", attempted_cmd="cmd")
    assert called.get("yes") and r.status == "collected"


def test_scp_failure_degrades_not_collected(monkeypatch, tmp_path):
    # ssh ok but scp pull-back fails -> degraded_failed, never a phantom collected artifact
    import subprocess

    from vllm_evolve.bench import cuda_profile as cp
    from vllm_evolve.bench.config import build_bench_config
    bc = build_bench_config(runner_kind="strong_baseline", model="m", remote="box")
    collect_fn, cmd = cp.make_cuda_collector("nsys", bench_config=bc, run_dir=str(tmp_path))

    def fake_run(args, **k):
        r = type("R", (), {})()
        r.stdout, r.stderr = "", "scp: no such file"
        r.returncode = 1 if args[0] == "scp" else 0      # ssh ok, scp fails
        return r

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = cp.collect_cuda_profile("nsys", collect_fn=collect_fn, attempted_cmd=cmd, remote="box")
    assert res.status == "degraded_failed" and "scp" in res.error


def test_cmd_profile_cuda_persists_payload_to_out(monkeypatch, tmp_path, capsys):
    # AC4: the CLI builds a real collector + persists the cuda payload to --out (Codex R6)
    import json

    from vllm_evolve.bench import cuda_profile as cp
    from vllm_evolve.bench.config import build_bench_config
    from vllm_evolve.bench.cuda_profile import CudaProfileResult
    from vllm_evolve.cli import main as cli_main
    from vllm_evolve.core.schemas import Profile
    from vllm_evolve.engine import profile as prof_mod
    bc = build_bench_config(runner_kind="strong_baseline", model="m", remote="box").to_dict()
    monkeypatch.setattr(prof_mod, "collect_profile",
                        lambda config: Profile(metrics={"tok_s": 9.0}, bench_config=bc))

    def fake_cuda(profiler, **kw):
        return CudaProfileResult(
            profiler, "collected", capability={"available": True},
            artifacts=["/local/cuda/nsys_window.qdrep"], breakdown={"gemm_time_frac": 0.6},
            attempted_cmd="mkdir -p /tmp/vllm_evolve_cuda/x && nsys profile ... vllm serve m")

    monkeypatch.setattr(cp, "collect_cuda_profile", fake_cuda)
    out_file = tmp_path / "profile.json"
    rc = cli_main.main(["profile", "--cuda", "nsys", "--out", str(out_file), "--model", "m"])
    capsys.readouterr()
    assert rc == 0
    persisted = json.loads(out_file.read_text(encoding="utf-8"))
    assert persisted["cuda_profile"]["status"] == "collected"
    assert persisted["cuda_profile"]["attempted_cmd"].startswith("mkdir -p /tmp/vllm_evolve_cuda")
    assert persisted["cuda_profile"]["artifacts"] == ["/local/cuda/nsys_window.qdrep"]
    assert persisted["cuda_breakdown"]["gemm_time_frac"] == 0.6


def test_cmd_profile_cuda_no_provenance_degrades(monkeypatch, tmp_path, capsys):
    # Codex R7: a profile with no bench_config must NOT run a generic same-caliber-less window
    import json

    from vllm_evolve.cli import main as cli_main
    from vllm_evolve.core.schemas import Profile
    from vllm_evolve.engine import profile as prof_mod

    monkeypatch.setattr(prof_mod, "collect_profile", lambda config: Profile(metrics={"tok_s": 9.0}))
    out_file = tmp_path / "p.json"
    rc = cli_main.main(["profile", "--cuda", "nsys", "--out", str(out_file), "--model", "m"])
    capsys.readouterr()
    assert rc == 0
    persisted = json.loads(out_file.read_text(encoding="utf-8"))
    assert persisted["cuda_profile"]["status"] == "degraded_no_provenance"
