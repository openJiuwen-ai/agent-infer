# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Load Inferact replay E2E cases and build vLLM serve / bench argv."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from .....helpers.case_loader import (
    BenchmarkCase,
    E2EConfig,
    PerfScenario,
    _fetch_hardware_slug_from_platform,
    _hardware_slug_from_mark,
    _make_run_tag,
    _repo_root,
    ensure_directory,
    load_case_file,
)

DEFAULT_BENCH_CONFIG = "agentinfer/agentbench/configs/replay_inferact.yaml"
_HF_DATASET_HINT = "https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces"


def resolve_replay_trace_path(benchmark: dict[str, object], *, repo_root: Path) -> Path:
    """Return the full trace file path for ``--trace-path`` from ``benchmark_params``."""

    raw = benchmark.get("trace-path")
    if not isinstance(raw, str) or not raw.strip():
        raise FileNotFoundError(
            f"benchmark_params.trace-path is required (e.g. downloaded codex_swebenchpro.json from {_HF_DATASET_HINT})"
        )
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"replay trace file not found: {path}")
    return path


@dataclass(frozen=True)
class ReplayE2EConfig:
    """Runtime configuration for one Inferact replay functional E2E case."""

    case: BenchmarkCase
    host: str
    port: int
    bench_config: Path
    trace_path: Path
    model: str
    task_num: int
    max_concurrency: int
    task_timeout_seconds: int
    result_root: Path
    vllm_log_dir: Path
    run_tag: str
    wait_for_vllm_ready: bool

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @classmethod
    def from_case_file(cls, case_file: Path) -> ReplayE2EConfig:
        case = load_case_file(case_file)
        if case.scenario is not PerfScenario.BASELINE:
            raise ValueError(f"replay E2E cases must use scenario baseline: {case_file}")
        benchmark = case.benchmark_params
        if benchmark.get("workload") != "inferact-replay":
            raise ValueError(f"benchmark_params.workload must be inferact-replay: {case_file}")

        repo = _repo_root()
        config_path = Path(str(benchmark.get("config", DEFAULT_BENCH_CONFIG)))
        if not config_path.is_absolute():
            config_path = (repo / config_path).resolve()

        result_root = case.deploy.result_root
        if not result_root.is_absolute():
            result_root = (repo / result_root).resolve()

        trace_path = resolve_replay_trace_path(benchmark, repo_root=repo)

        run_tag = _make_run_tag()
        return cls(
            case=case,
            host=str(benchmark.get("host", "127.0.0.1")),
            port=int(benchmark.get("port", 8077)),
            bench_config=config_path,
            trace_path=trace_path,
            model=str(benchmark.get("model", "glm-5")),
            task_num=int(benchmark.get("task-num", 8)),
            max_concurrency=int(benchmark.get("max-concurrency", 4)),
            task_timeout_seconds=int(benchmark.get("timeout", 3600)),
            result_root=result_root,
            vllm_log_dir=result_root / "vllm-logs",
            run_tag=run_tag,
            wait_for_vllm_ready=case.wait_for_vllm_ready,
        )

    def with_benchmark_load(
        self, *, task_num: int | None = None, max_concurrency: int | None = None
    ) -> ReplayE2EConfig:
        updates: dict[str, int] = {}
        if task_num is not None:
            updates["task_num"] = task_num
        if max_concurrency is not None:
            updates["max_concurrency"] = max_concurrency
        return replace(self, **updates) if updates else self

    def with_fresh_run(self) -> ReplayE2EConfig:
        """Return a copy with a new run tag for a second cold vLLM + replay cycle."""

        return replace(self, run_tag=_make_run_tag())

    def make_result_dir(self, *, hardware_slug: str) -> Path:
        if hardware_slug == "unknown":
            detected = _fetch_hardware_slug_from_platform()
            if detected:
                hardware_slug = detected
            else:
                mark_slug = _hardware_slug_from_mark(self.case.mark)
                if mark_slug:
                    hardware_slug = mark_slug
        host_slug = "local" if self.host in {"127.0.0.1", "localhost"} else self.host.replace(".", "-")
        name = f"run-{hardware_slug}-{host_slug}-replay-{self.task_num}-{self.max_concurrency}-{self.run_tag}"
        ensure_directory(
            self.result_root,
            run_as_user=self.to_e2e_config().effective_benchmark_run_as_user(),
        )
        return self.result_root / name

    def validate_prerequisites(self) -> list[str]:
        blockers: list[str] = []
        if not self.case.deploy.model.exists():
            blockers.append(f"model path does not exist: {self.case.deploy.model}")
        import shutil

        if not shutil.which("vllm"):
            blockers.append("vllm executable is unavailable on PATH")
        if not self.bench_config.is_file():
            blockers.append(f"bench config does not exist: {self.bench_config}")
        if not self.trace_path.is_file():
            blockers.append(f"trace path does not exist: {self.trace_path}")
        return blockers

    def to_e2e_config(self) -> E2EConfig:
        """Adapt to shared ``E2EConfig`` for vLLM lifecycle helpers."""

        e2e = E2EConfig.from_case(self.case)
        return replace(
            e2e,
            host=self.host,
            port=self.port,
            bench_config=self.bench_config,
            task_num=self.task_num,
            max_concurrency=self.max_concurrency,
            task_timeout_seconds=self.task_timeout_seconds,
            result_root=self.result_root,
            vllm_log_dir=self.vllm_log_dir,
            run_tag=self.run_tag,
            wait_for_vllm_ready=self.wait_for_vllm_ready,
            agent_profile="single",
        )


def build_replay_argv(config: ReplayE2EConfig, result_dir: Path) -> list[str]:
    """Build replay CLI; inferact defaults live in ``replay_inferact.yaml``."""

    argv = [
        "vllm",
        "bench",
        "serve",
        "--agentinfer",
        "replay",
        "--config",
        str(config.bench_config),
        "--result-dir",
        str(result_dir),
        "--trace-path",
        str(config.trace_path),
        "--model",
        config.model,
        "--task-num",
        str(config.task_num),
        "--max-concurrency",
        str(config.max_concurrency),
        "--base-url",
        config.base_url,
    ]
    metrics_url = config.case.benchmark_params.get("metrics-url")
    if isinstance(metrics_url, str) and metrics_url.strip():
        argv.extend(["--metrics-url", metrics_url.strip()])
    return argv


def build_replay_env() -> dict[str, str]:
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["TQDM_DISABLE"] = "1"
    return env
