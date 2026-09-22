#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Run resumable TP4 serving experiments and append evidence to the RSI ledger.

The sweep keeps the trace, seed, replay concurrency and exact calibration
fixed. It changes one vLLM serving profile at a time, warms each profile once,
and records five measured repetitions before moving to the next profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from agentinfer.rsi.experiments import append_experiment, list_experiments

DEFAULT_MODEL = "/home/zjy/code/david/b_workspace/models/Qwen3.8-27B"
DEFAULT_TRACE = "/home/zjy/code/hsliu/tmp/rsi-pr25-b300-20260922/codex-trace/codex_swebenchpro.json"
DEFAULT_CONVERTED = "/home/zjy/code/hsliu/tmp/rsi-pr25-b300-20260922/results/replay-tp4-smoke/convert_result"
DEFAULT_RUN_DIR = "/home/zjy/code/hsliu/tmp/rsi-pr25-b300-20260922/rsi-real"
DEFAULT_SWEEP_DIR = "/home/zjy/code/hsliu/tmp/rsi-pr25-b300-20260922/results/rsi-50-rounds"
DEFAULT_PORT = 8000
DEFAULT_SEED = 228
REPETITIONS = 5


@dataclass(frozen=True)
class Profile:
    name: str
    hypothesis: str
    change: str
    server_args: tuple[str, ...]


PROFILES = (
    Profile(
        "baseline-current",
        "The existing I2 TP4 serving profile remains stable across repeated warm runs.",
        "Repeat the I2 server arguments without changing a serving knob.",
        (),
    ),
    Profile(
        "no-prefix-cache",
        "Prefix caching helps this continuation-heavy trace more than its lookup overhead costs.",
        "Disable prefix caching while keeping the model, TP4, scheduler and cache budget fixed.",
        ("--no-enable-prefix-caching",),
    ),
    Profile(
        "batch-8192",
        "A smaller token budget per scheduler step may reduce long-prefill interference with decode.",
        "Set --max-num-batched-tokens=8192; keep chunked prefill enabled by vLLM defaults.",
        ("--max-num-batched-tokens", "8192"),
    ),
    Profile(
        "batch-32768",
        "A larger token budget per scheduler step may improve long-prefill efficiency.",
        "Set --max-num-batched-tokens=32768.",
        ("--max-num-batched-tokens", "32768"),
    ),
    Profile(
        "batch-65536",
        "A still larger scheduler batch may increase aggregate prefill throughput on B300.",
        "Set --max-num-batched-tokens=65536.",
        ("--max-num-batched-tokens", "65536"),
    ),
    Profile(
        "async-on",
        "Explicit asynchronous scheduling reduces host-side gaps between engine steps.",
        "Enable --async-scheduling explicitly.",
        ("--async-scheduling",),
    ),
    Profile(
        "async-off",
        "Synchronous scheduling is a negative control for the asynchronous scheduler hypothesis.",
        "Disable --async-scheduling explicitly.",
        ("--no-async-scheduling",),
    ),
    Profile(
        "stream-8",
        "Larger streaming batches may reduce host response overhead without changing generated tokens.",
        "Set --stream-interval=8.",
        ("--stream-interval", "8"),
    ),
    Profile(
        "max-seqs-8",
        "Allowing more sequences per scheduler step may improve overlap when request arrival bursts.",
        "Set --max-num-seqs=8 while the replay client remains at concurrency two.",
        ("--max-num-seqs", "8"),
    ),
    Profile(
        "kv-fp8",
        "FP8 KV storage may reduce KV bandwidth and cache pressure on B300 without changing weights.",
        "Set --kv-cache-dtype=fp8_e4m3; the candidate must pass GSM8K before promotion.",
        ("--kv-cache-dtype", "fp8_e4m3"),
    ),
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, str | None]:
    return {"path": str(path), "sha256": sha256(path)}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_ready(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> float:
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness with code {process.returncode}")
        try:
            with urlopen(f"{base_url}/health", timeout=5) as response:  # noqa: S310 - loopback URL
                if response.status == 200:
                    return time.monotonic() - started
        except (OSError, URLError):
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready within {timeout:g}s")


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=90)
    except (KeyboardInterrupt, ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            pass


def write_config(
    path: Path,
    *,
    result_dir: Path,
    base_url: str,
    trace_path: Path,
    converted_trace_path: Path,
    seed: int,
    task_num: int,
) -> None:
    text = f"""experiment:
  task_num: {task_num}
  result_dir: {result_dir}
  max_concurrency: 2
  task_timeout_seconds: 1800
  run_timeout_seconds: 2400

backend:
  type: vllm
  base_url: {base_url}
  metrics_url: {base_url}/metrics
  tokenizer_base_url: {base_url}
  model: Qwen/Qwen3.8-27B
  endpoint: /v1/chat/completions
  chat_template_kwargs: {{}}
  api_key_env: null

replay:
  trace_type: inferact_codex_swebenchpro
  trace_path: {trace_path}
  converted_trace_path: {converted_trace_path}
  interval_mode: lognormal
  interval_lognormal:
    p50_seconds: 0.001
    p95_seconds: 0.002278
    p99_seconds: 0.003205
  sample_seed: {seed}
  prompt_shape: trace_record
  max_input_tokens: 240000
  max_output_tokens: null
  context_adjustment_mode: adaptive
  context_micro_trim_max_tokens: 64
  context_micro_trim_max_ratio: 0.005
  prompt_calibration_tolerance_tokens: 0
  request_timeout_seconds: 300
"""
    path.write_text(text, encoding="utf-8")


def run_replay(python: str, config: Path, log_path: Path, repo: Path) -> int:
    dispatcher = (
        "import sys; "
        "from agentinfer.agentcache.entrypoints.cli.main import main; "
        "raise SystemExit(main(['vllm', *sys.argv[1:]]))"
    )
    command = [
        python,
        "-c",
        dispatcher,
        "bench",
        "serve",
        "--agentinfer",
        "replay",
        "--config",
        str(config),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=3600,
            check=False,
        )
    return completed.returncode


def summarize(result_dir: Path) -> tuple[dict[str, object], str | None]:
    summary_path = result_dir / "summary.json"
    execution_path = result_dir / "replay-execution.json"
    if not summary_path.is_file():
        return {}, "Replay did not produce summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    requests = summary.get("requests", {})
    tasks = summary.get("tasks", {})
    execution = summary.get("execution", {}).get("metadata", {})
    lifecycle = summary.get("lifecycle", {})
    ttft_p50 = requests.get("ttft_seconds", {}).get("p50")
    planned = execution.get("planned_requests")
    successful = requests.get("successful_requests")
    failure = None
    if lifecycle.get("status") != "completed":
        failure = f"Replay lifecycle status: {lifecycle.get('status')}: {lifecycle.get('error')}"
    elif planned != successful or requests.get("failed_requests", 0) != 0:
        failure = (
            f"Incomplete replay coverage: planned={planned}, successful={successful}, "
            f"failed={requests.get('failed_requests')}"
        )
    metrics = {
        "throughput_output_tokens_per_s": summary.get("output_token_throughput_per_second"),
        "throughput_input_tokens_per_s": summary.get("input_token_throughput_per_second"),
        "throughput_requests_per_s": summary.get("request_throughput_per_second"),
        "ttft_p50_ms": None if ttft_p50 is None else ttft_p50 * 1000,
        "replay_planned_tasks": tasks.get("completed", 0) + tasks.get("failed", 0),
        "replay_completed_tasks": tasks.get("completed"),
        "replay_failed_tasks": tasks.get("failed"),
        "replay_exact_input_requests": execution.get("backend_input_checked_requests"),
        "replay_successful_requests": successful,
        "replay_failed_requests": requests.get("failed_requests"),
        "replay_calibration_max_residual_tokens": execution.get("prompt_calibration_max_absolute_residual_tokens"),
        "prefix_cache_hit_rate": summary.get("vllm", {}).get("prefix_cache_hit_rate"),
        "gsm8k_accuracy": None,
        "tpot_p50_ms": None,
        "readiness_seconds": None,
    }
    if not execution_path.is_file():
        failure = failure or "Replay did not produce replay-execution.json"
    return metrics, failure


def make_record(
    *,
    round_id: str,
    profile: Profile,
    repeat: int,
    result_dir: Path,
    config_path: Path,
    replay_log: Path,
    server_log: Path,
    metrics: dict[str, object],
    failure: str | None,
    readiness_seconds: float | None,
    commit: str,
) -> dict[str, object]:
    metrics = dict(metrics)
    metrics["readiness_seconds"] = readiness_seconds
    status = "measured" if failure is None else "failed"
    return {
        "round_id": round_id,
        "timestamp": timestamp(),
        "hypothesis": profile.hypothesis,
        "change": f"{profile.change} Repetition {repeat}/{REPETITIONS}.",
        "model": "Qwen/Qwen3.8-27B",
        "backend": f"cuda/vllm-tp4/{profile.name}",
        "component_version": f"agentinfer-{commit[:12]}+vllm-sweep",
        "config": {
            "precision": "bf16",
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "max_concurrency": 2,
            "trace_seed": DEFAULT_SEED,
            "trace_mode": "inferact_codex_swebenchpro",
            "prompt_calibration_tolerance_tokens": 0,
            "prefix_caching": "--no-enable-prefix-caching" not in profile.server_args,
            "server_args": list(profile.server_args),
            "repetition": repeat,
        },
        "commands": [
            "vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 ...",
            f"{sys.executable} -c agentinfer.agentcache.entrypoints.cli.main "
            f"bench serve --agentinfer replay --config {config_path}",
        ],
        "metrics": metrics,
        "status": status,
        "evidence": [
            evidence(config_path),
            evidence(result_dir / "summary.json"),
            evidence(result_dir / "replay-execution.json"),
            evidence(replay_log),
            evidence(server_log),
        ],
        "failure_reason": failure,
        "next_test": "Compare profile medians and run full GSM8K on the best valid profile.",
        "baseline_round_id": "I2-tp4-seed228-exact",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_RUN_DIR))
    parser.add_argument("--sweep-dir", type=Path, default=Path(DEFAULT_SWEEP_DIR))
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--trace", type=Path, default=Path(DEFAULT_TRACE))
    parser.add_argument("--converted-trace", type=Path, default=Path(DEFAULT_CONVERTED))
    parser.add_argument("--vllm", default=shutil.which("vllm") or "vllm")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--readiness-timeout", type=float, default=1800)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds <= 0 or args.rounds > len(PROFILES) * REPETITIONS:
        raise SystemExit(f"--rounds must be between 1 and {len(PROFILES) * REPETITIONS}")
    args.sweep_dir.mkdir(parents=True, exist_ok=True)
    records = list_experiments(args.run_dir)
    existing = {str(record["round_id"]) for record in records}
    numbered = [
        int(str(record["round_id"])[1:])
        for record in records
        if str(record["round_id"]).startswith("I") and str(record["round_id"])[1:].isdigit()
    ]
    next_number = max(numbered, default=4) + 1
    remaining = args.rounds
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    base_url = f"http://127.0.0.1:{args.port}"
    for profile_index, profile in enumerate(PROFILES):
        if remaining <= 0:
            break
        profile_rounds = min(REPETITIONS, remaining)
        profile_dir = args.sweep_dir / profile.name
        profile_dir.mkdir(parents=True, exist_ok=True)
        server_log = profile_dir / "server.log"
        server_command = [
            args.vllm,
            "serve",
            str(args.model),
            "--served-model-name",
            "Qwen/Qwen3.8-27B",
            "--tensor-parallel-size",
            "4",
            "--max-model-len",
            "262144",
            "--gpu-memory-utilization",
            "0.90",
            "--enable-prefix-caching",
            "--enable-tokenizer-info-endpoint",
            "--generation-config",
            "vllm",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            *profile.server_args,
        ]
        with server_log.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(server_command) + "\n")
        process: subprocess.Popen[bytes] | None = None
        readiness: float | None = None
        pending: list[dict[str, object]] = []
        try:
            server_stream = server_log.open("ab")
            process = subprocess.Popen(
                server_command,
                cwd=args.repo,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
                stdout=server_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            readiness = wait_ready(base_url, process, args.readiness_timeout)
            if not args.no_warmup:
                warmup_result = profile_dir / "warmup"
                warmup_config = profile_dir / "warmup.yaml"
                warmup_log = profile_dir / "warmup.log"
                write_config(
                    warmup_config,
                    result_dir=warmup_result,
                    base_url=base_url,
                    trace_path=args.trace,
                    converted_trace_path=args.converted_trace,
                    seed=DEFAULT_SEED,
                    task_num=2,
                )
                run_replay(sys.executable, warmup_config, warmup_log, args.repo)
            for repeat in range(1, profile_rounds + 1):
                round_number = next_number + len(pending)
                round_id = f"I{round_number}"
                if round_id in existing:
                    continue
                result_dir = profile_dir / f"repeat-{repeat}"
                config_path = profile_dir / f"repeat-{repeat}.yaml"
                replay_log = profile_dir / f"repeat-{repeat}.log"
                write_config(
                    config_path,
                    result_dir=result_dir,
                    base_url=base_url,
                    trace_path=args.trace,
                    converted_trace_path=args.converted_trace,
                    seed=DEFAULT_SEED,
                    task_num=2,
                )
                return_code = run_replay(sys.executable, config_path, replay_log, args.repo)
                metrics, failure = summarize(result_dir)
                if return_code != 0:
                    failure = failure or f"Replay command exited with code {return_code}"
                pending.append(
                    make_record(
                        round_id=round_id,
                        profile=profile,
                        repeat=repeat,
                        result_dir=result_dir,
                        config_path=config_path,
                        replay_log=replay_log,
                        server_log=server_log,
                        metrics=metrics,
                        failure=failure,
                        readiness_seconds=readiness,
                        commit=commit,
                    )
                )
                print(
                    f"{round_id} {profile.name} repeat={repeat}/{profile_rounds} "
                    f"status={'measured' if failure is None else 'failed'} "
                    f"output_tok_s={metrics.get('throughput_output_tokens_per_s')}",
                    flush=True,
                )
        except Exception as error:  # record every planned round, then continue to the next profile
            for repeat in range(len(pending) + 1, profile_rounds + 1):
                round_id = f"I{next_number + len(pending)}"
                result_dir = profile_dir / f"repeat-{repeat}"
                config_path = profile_dir / f"repeat-{repeat}.yaml"
                write_config(
                    config_path,
                    result_dir=result_dir,
                    base_url=base_url,
                    trace_path=args.trace,
                    converted_trace_path=args.converted_trace,
                    seed=DEFAULT_SEED,
                    task_num=2,
                )
                pending.append(
                    make_record(
                        round_id=round_id,
                        profile=profile,
                        repeat=repeat,
                        result_dir=result_dir,
                        config_path=config_path,
                        replay_log=profile_dir / f"repeat-{repeat}.log",
                        server_log=server_log,
                        metrics={"gsm8k_accuracy": None},
                        failure=f"Profile execution error: {type(error).__name__}: {error}",
                        readiness_seconds=readiness,
                        commit=commit,
                    )
                )
            print(f"{profile.name} failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        finally:
            stop_process(process)
            if process is not None:
                server_stream.close()
        for record in pending:
            append_experiment(args.run_dir, record)
            existing.add(str(record["round_id"]))
            print(f"appended {record['round_id']} {record['status']}", flush=True)
        next_number += len(pending)
        remaining -= len(pending)
        if remaining <= 0:
            break
        print(f"completed profile {profile_index + 1}/{len(PROFILES)}; remaining={remaining}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
