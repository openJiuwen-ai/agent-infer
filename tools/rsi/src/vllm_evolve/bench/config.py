"""Layered ``BenchConfig`` — the single source of truth for one benchmark run.

Split into ``EngineConfig`` / ``WorkloadConfig`` / ``RunnerConfig`` /
``StatisticalConfig`` / ``EnvironmentConfig`` so every facet of a run's *口径*
lives in exactly one place and is snapshotted into ``eval_result`` for provenance.
Renders to ``vllm serve`` args via a single entry point, so a **vanilla** run
(vLLM default scheduler, no plugin) and a **candidate** run (our scheduler plugin)
differ ONLY by ``runner_kind`` / ``policy`` — never by an accidental knob drift.

See ``.humanize/plans/e2e-optimization-plan.md`` M-B1 (解 GAP-B / GAP-CONFIG).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

VANILLA = "vanilla"              # vLLM default scheduler + default knobs — the naive baseline
STRONG_BASELINE = "strong_baseline"  # vLLM default scheduler + ALL known optimizations tuned
CANDIDATE = "candidate"          # our scheduler plugin / new algorithm attached
_BASELINE_KINDS = (VANILLA, STRONG_BASELINE)   # neither attaches our plugin
# artifact kinds whose weights live in a separate quantized checkpoint served in place
_ARTIFACT_KINDS = ("awq-ckpt", "gptq-ckpt", "fp8-ckpt")

# THE single source of truth for executable engine levers: the EngineConfig fields that
# ``to_serve_args()`` renders AND native/dispatch accept. autopt.levers and autopt.profile
# import this so selection/search/profiling never diverges from the executed bench surface
# (Codex R1). Workload search dims (concurrency/n_requests) extend it in profile.WIRED_KNOBS.
WIRED_LEVERS = frozenset({
    "quantization", "kv_cache_dtype", "max_num_batched_tokens", "max_num_seqs",
    "gpu_memory_utilization", "enable_chunked_prefill", "enable_prefix_caching",
    "tensor_parallel_size", "max_model_len", "enforce_eager",
})


@dataclass
class EngineConfig:
    """Everything that affects the vLLM *engine* (model representation + serve knobs)."""
    model: str = "facebook/opt-125m"
    model_artifact_kind: str = "hf-fp16"        # hf-fp16 | awq-ckpt | fp8-ckpt | gptq-ckpt
    quantized_model_id: str | None = None        # path/id of a quantized checkpoint (if any)
    weight_dtype: str | None = None              # None = vLLM default (fp16/bf16)
    kv_cache_dtype: str | None = None            # None = auto | "fp8"
    quantization: str | None = None              # None | "fp8" | "awq" | "gptq"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int | None = None
    enable_prefix_caching: bool | None = None    # None = leave vLLM default
    enable_chunked_prefill: bool | None = None   # None = leave vLLM default
    enforce_eager: bool = False
    async_scheduling: bool | None = None         # None = auto; explicit bool is rendered
    scheduler_cls: str | None = None             # None = vanilla; set for candidate
    served_name: str = "ve_bench"
    vllm_version_pin: str | None = None


@dataclass
class WorkloadConfig:
    """Everything that defines the *load* presented to the server."""
    regime: str = "throughput"
    load_mode: str = "closed"                    # closed (concurrency) | open (arrival rate)
    concurrency: int | None = None
    arrival_rate_qps: float | None = None
    n_requests: int = 200
    warmup: int = 0
    trace_path: str | None = None                 # materialized real-vLLM workload JSON
    workload_spec: dict = field(default_factory=dict)   # heterogeneous-load params
    request_generator_version: str = "v1"
    tokenizer: str | None = None
    token_accounting: str = "tokenizer"          # tokenizer (real count) | server-reported


@dataclass
class RunnerConfig:
    """How/where the run is executed (vanilla vs candidate, remote box).

    The remote-environment fields default from the SAME ``VE_*`` env vars as ``dispatch.DEFAULTS``,
    so a default-built BenchConfig records the user's configured environment in provenance AND the
    SSH run executes there (Codex review P2): forwarding these to the bench must not silently
    override the documented ``VE_REMOTE_REPO`` / ``VE_CONDA_ENV`` / ``VE_HF_ENDPOINT`` overrides.
    """
    runner_kind: str = CANDIDATE                 # vanilla | candidate
    policy_path: str | None = None               # candidate policy .py (None for vanilla)
    port: int = 8200
    remote: str = field(default_factory=lambda: os.environ.get("VE_REMOTE", "gpu-host"))
    remote_repo: str = field(
        default_factory=lambda: os.environ.get("VE_REMOTE_REPO", "/workspace/vllm-evolve"))
    conda_env: str = field(default_factory=lambda: os.environ.get("VE_CONDA_ENV", "vllm-evolve"))
    conda_sh: str = field(default_factory=lambda: os.environ.get(
        "VE_CONDA_SH", "/workspace/vllm-evolve/miniforge3/etc/profile.d/conda.sh"))
    remote_workspace: str = field(default_factory=lambda: os.environ.get(
        "VE_REMOTE_WORKSPACE", "/workspace/vllm-evolve"))
    local_artifact_root: str = field(default_factory=lambda: os.environ.get(
        "VE_LOCAL_ARTIFACT_ROOT", "runs/real_vllm"))
    hf_endpoint: str = field(
        default_factory=lambda: os.environ.get("VE_HF_ENDPOINT", "https://hf-mirror.com"))


@dataclass
class StatisticalConfig:
    """Metric + significance gating (acceptance is paired-bootstrap, not point-only)."""
    primary_metric: str = "output_throughput_tok_s"
    slo: dict = field(default_factory=dict)
    seed_tiers: list = field(default_factory=lambda: [0, 1, 2])
    cv_threshold: float = 0.10
    n_boot: int = 2000
    epsilon_pct: float = 2.0
    accept_threshold_pct: float = 20.0           # X% hard gate vs the STRONG baseline
    timeout_s: int = 1200
    max_seeds: int | None = None                 # cap seeds run on the box (budget/smoke)


@dataclass
class EnvironmentConfig:
    gpus: str = "0"
    # ``auto`` is resolved once into a concrete ``gpus`` list before the first
    # real run.  The evidence remains in every cloned candidate config so an
    # evolution round cannot drift across physical cards.
    gpu_selection_mode: str = "fixed"
    gpu_selection_evidence: dict = field(default_factory=dict)
    vllm_version: str | None = None
    driver: str | None = None
    container_id: str | None = None


@dataclass
class BenchConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # ------------------------------------------------------------------ render
    def is_vanilla(self) -> bool:
        return self.runner.runner_kind == VANILLA

    def is_baseline(self) -> bool:
        """vanilla OR strong_baseline — neither attaches our scheduler plugin."""
        return self.runner.runner_kind in _BASELINE_KINDS

    def model_to_serve(self) -> str:
        """Model id/path actually passed to ``vllm serve``. Artifact-based weight
        quantization (awq/gptq/fp8 checkpoints) serves the quantized artifact in place
        of the base model; everything else serves the base model id."""
        e = self.engine
        if e.model_artifact_kind in _ARTIFACT_KINDS and e.quantized_model_id:
            return e.quantized_model_id
        return e.model

    def to_serve_args(self) -> list[str]:
        """Render ``vllm serve`` flags. A vanilla run NEVER gets ``--scheduler-cls``."""
        e = self.engine
        args: list[str] = [
            "--served-model-name", e.served_name,
            "--gpu-memory-utilization", str(e.gpu_memory_utilization),
            "--max-model-len", str(e.max_model_len),
            "--max-num-seqs", str(e.max_num_seqs),
        ]
        if e.tensor_parallel_size and e.tensor_parallel_size > 1:
            args += ["--tensor-parallel-size", str(e.tensor_parallel_size)]
        if e.max_num_batched_tokens:
            args += ["--max-num-batched-tokens", str(e.max_num_batched_tokens)]
        if e.quantization:
            args += ["--quantization", e.quantization]
        if e.kv_cache_dtype:
            args += ["--kv-cache-dtype", e.kv_cache_dtype]
        if e.enable_prefix_caching is True:
            args += ["--enable-prefix-caching"]
        elif e.enable_prefix_caching is False:
            args += ["--no-enable-prefix-caching"]
        if e.enable_chunked_prefill is True:
            args += ["--enable-chunked-prefill"]
        elif e.enable_chunked_prefill is False:
            args += ["--no-enable-chunked-prefill"]    # explicit disable (False != None=default)
        if e.async_scheduling is True:
            args += ["--async-scheduling"]
        elif e.async_scheduling is False:
            args += ["--no-async-scheduling"]
        if e.enforce_eager:
            args += ["--enforce-eager"]
        # candidate ONLY: attach our scheduler plugin. vanilla omits it (GAP-B).
        if self.runner.runner_kind == CANDIDATE and e.scheduler_cls:
            args += ["--scheduler-cls", e.scheduler_cls]
        return args

    # ------------------------------------------------------------- provenance
    def to_dict(self) -> dict:
        return asdict(self)

    def provenance(self) -> dict:
        """Snapshot written into eval_result: full config + the resolved scheduler facts."""
        return {
            "bench_config": self.to_dict(),
            "model_served": self.model_to_serve(),
            "rendered_serve_args": self.to_serve_args(),
            "runner_kind": self.runner.runner_kind,
            "is_baseline": self.is_baseline(),
            "scheduler_cls_requested": None if self.is_baseline() else self.engine.scheduler_cls,
            "async_scheduling": self.engine.async_scheduling,
            "vllm_version_pin": self.engine.vllm_version_pin,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BenchConfig:
        def sub(key, klass):
            v = d.get(key) or {}
            return klass(**{k: vv for k, vv in v.items() if k in klass.__dataclass_fields__})
        return cls(engine=sub("engine", EngineConfig), workload=sub("workload", WorkloadConfig),
                   runner=sub("runner", RunnerConfig),
                   statistical=sub("statistical", StatisticalConfig),
                   environment=sub("environment", EnvironmentConfig))


def same_caliber(a: BenchConfig, b: BenchConfig, *,
                 exempt: set[str] | None = None) -> tuple[bool, list[str]]:
    """AC1 check: two configs share the SAME 口径 except runner_kind/policy/scheduler_cls.

    Returns (ok, diffs). Used to prove a vanilla/candidate A/B differs only by the
    thing under test — no accidental engine/workload drift. ``exempt`` names the engine/workload
    knob(s) that ARE the thing under test (e.g. an autopt config-optimization searched
    ``max_num_seqs``): those may legitimately differ, so they are skipped — without exempting them,
    every real config-optimization A/B would falsely read as same_caliber_mismatch. Any OTHER
    engine/workload drift still trips a mismatch.
    """
    exempt = exempt or set()
    diffs: list[str] = []
    ea, eb = asdict(a.engine), asdict(b.engine)
    for k in ea:
        if k in ("scheduler_cls",) or k in exempt:
            continue
        if ea[k] != eb[k]:
            diffs.append(f"engine.{k}: {ea[k]} != {eb[k]}")
    wa, wb = asdict(a.workload), asdict(b.workload)
    for k in wa:
        if k in exempt:
            continue
        if wa[k] != wb[k]:
            diffs.append(f"workload.{k}: {wa[k]} != {wb[k]}")
    if asdict(a.statistical) != asdict(b.statistical):
        diffs.append("statistical configs differ")
    # Runner: only runner_kind/policy_path/port may differ; remote/env/conda must match,
    # else the A/B ran on different machines/setups and is not comparable (Codex HIGH#2).
    ra, rb = asdict(a.runner), asdict(b.runner)
    for k in ra:
        if k in ("runner_kind", "policy_path", "port"):
            continue
        if ra[k] != rb[k]:
            diffs.append(f"runner.{k}: {ra[k]} != {rb[k]}")
    # Environment (GPU / driver / vLLM version / container) must be identical.
    na, nb = asdict(a.environment), asdict(b.environment)
    for k in na:
        if na[k] != nb[k]:
            diffs.append(f"environment.{k}: {na[k]} != {nb[k]}")
    return (not diffs), diffs


def build_bench_config(*, runner_kind: str = CANDIDATE, policy_path: str | None = None,
                       model: str = "facebook/opt-125m", scheduler_cls: str | None = None,
                       profile: str = "throughput", gpus: str = "0",
                       remote: str | None = None, port: int = 8200,
                       **overrides) -> BenchConfig:
    """Single construction point for ``ve bench`` / ``ve calibrate`` — build a BenchConfig
    from flat CLI-style kwargs. Engine knobs (quantization, kv_cache_dtype,
    quantized_model_id, model_artifact_kind, max_num_batched_tokens, enable_chunked_prefill,
    ...) AND workload knobs (concurrency, n_requests, load_mode, warmup, arrival_rate_qps)
    are split to the right sub-config so the sweep actually varies the workload (Codex R2).
    """
    # max_seeds is a run-cardinality cap (not an engine/workload knob) -> StatisticalConfig, so a
    # --max-seeds budget reaches run_remote_bench instead of being filtered away (Codex P2).
    max_seeds = overrides.pop("max_seeds", None)
    eng_fields = EngineConfig.__dataclass_fields__
    wl_fields = WorkloadConfig.__dataclass_fields__
    wl_kw = {k: v for k, v in overrides.items() if k in wl_fields and v is not None}
    eng_kw = {k: v for k, v in overrides.items()
              if k in eng_fields and k not in wl_fields and v is not None}
    engine = EngineConfig(model=model, scheduler_cls=scheduler_cls, **eng_kw)
    # Only override RunnerConfig.remote when a remote was explicitly given; otherwise let its
    # VE_REMOTE-seeded default_factory run, so a default ve bench/calibrate run honors VE_REMOTE
    # AND records the same host in provenance (Codex review P2).
    runner_kw = {"runner_kind": runner_kind, "policy_path": policy_path, "port": port}
    if remote is not None:
        runner_kw["remote"] = remote
    runner = RunnerConfig(**runner_kw)
    return BenchConfig(engine=engine, workload=WorkloadConfig(regime=profile, **wl_kw),
                       runner=runner, environment=EnvironmentConfig(gpus=gpus),
                       statistical=StatisticalConfig(max_seeds=max_seeds))
