# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Load Trace Replay E2E cases and build vLLM serve / bench argv."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ....helpers.case_loader import (
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

TracePathMode = Literal["required", "optional"]


@dataclass(frozen=True)
class ReplaySuiteSpec:
    """Per-trace-family defaults and trace-path policy for replay functional E2E."""

    workload: str
    default_bench_config: str
    trace_path_mode: TracePathMode
    missing_trace_path_hint: str
    optional_trace_download_hint: str = ""


def resolve_replay_trace_path(
    benchmark: dict[str, object],
    *,
    repo_root: Path,
    spec: ReplaySuiteSpec,
) -> Path | None:
    """Return the trace file for ``--trace-path``, or ``None`` when optional and omitted."""

    raw = benchmark.get("trace-path")
    if spec.trace_path_mode == "optional":
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw.strip():
            return None
    elif not isinstance(raw, str) or not raw.strip():
        raise FileNotFoundError(spec.missing_trace_path_hint)

    assert isinstance(raw, str)
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
    """Runtime configuration for one Trace Replay functional E2E case."""

    suite: ReplaySuiteSpec
    case: BenchmarkCase
    host: str
    port: int
    bench_config: Path
    trace_path: Path | None
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
    def from_case_file(cls, case_file: Path, suite: ReplaySuiteSpec) -> ReplayE2EConfig:
        case = load_case_file(case_file)
        if case.scenario is not PerfScenario.BASELINE:
            raise ValueError(f"replay E2E cases must use scenario baseline: {case_file}")
        benchmark = case.benchmark_params
        if benchmark.get("workload") != suite.workload:
            raise ValueError(f"benchmark_params.workload must be {suite.workload}: {case_file}")

        repo = _repo_root()
        config_path = Path(str(benchmark.get("config", suite.default_bench_config)))
        if not config_path.is_absolute():
            config_path = (repo / config_path).resolve()

        result_root = case.deploy.result_root
        if not result_root.is_absolute():
            result_root = (repo / result_root).resolve()

        trace_path = resolve_replay_trace_path(benchmark, repo_root=repo, spec=suite)

        run_tag = _make_run_tag()
        return cls(
            suite=suite,
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
        self,
        *,
        task_num: int | None = None,
        max_concurrency: int | None = None,
        bench_config: str | Path | None = None,
    ) -> ReplayE2EConfig:
        updates: dict[str, object] = {}
        if task_num is not None:
            updates["task_num"] = task_num
        if max_concurrency is not None:
            updates["max_concurrency"] = max_concurrency
        if bench_config is not None:
            path = Path(bench_config).expanduser()
            if not path.is_absolute():
                path = (_repo_root() / path).resolve()
            else:
                path = path.resolve()
            updates["bench_config"] = path
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
        if self.trace_path is not None and not self.trace_path.is_file():
            blockers.append(f"trace path does not exist: {self.trace_path}")
        if self.suite.trace_path_mode == "optional" and self.trace_path is None and self.task_num < 1:
            blockers.append(
                f"trace-path omitted; task-num must be set for TraceLab download "
                f"({self.suite.optional_trace_download_hint})"
            )
        if self.suite.trace_path_mode == "required" and self.trace_path is None:
            blockers.append(self.suite.missing_trace_path_hint)
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


_BENCHMARK_REPLAY_CLI_FLAGS: tuple[tuple[str, str], ...] = (
    ("metrics-url", "--metrics-url"),
    ("endpoint", "--endpoint"),
    ("trace-type", "--trace-type"),
    ("interval-mode", "--interval-mode"),
)


def _append_benchmark_replay_cli(argv: list[str], benchmark: dict[str, object]) -> None:
    """Append optional Replay bench CLI flags mirrored from ``benchmark_params``."""

    for param_key, cli_flag in _BENCHMARK_REPLAY_CLI_FLAGS:
        raw = benchmark.get(param_key)
        if isinstance(raw, str) and raw.strip():
            argv.extend([cli_flag, raw.strip()])


def build_replay_argv(config: ReplayE2EConfig, result_dir: Path) -> list[str]:
    """Build ``vllm bench serve --agentinfer replay`` argv for one E2E run."""

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
        "--model",
        config.model,
        "--task-num",
        str(config.task_num),
        "--max-concurrency",
        str(config.max_concurrency),
        "--base-url",
        config.base_url,
    ]
    if config.trace_path is not None:
        argv.extend(["--trace-path", str(config.trace_path)])
    _append_benchmark_replay_cli(argv, config.case.benchmark_params)
    return argv


def build_replay_env() -> dict[str, str]:
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["TQDM_DISABLE"] = "1"
    return env
