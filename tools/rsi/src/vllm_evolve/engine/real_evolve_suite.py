"""Local controller for the three-scenario real-vLLM verification round.

Search and candidate source stay local.  The controller sends immutable inputs
to the remote worker, runs a smoke test, then paired baseline/candidate (and
optional mechanism control) measurements for BurstGPT, moderate stress, and
severe stress.  It writes progress after every run so interruption never erases
completed evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from vllm_evolve.bench.config import (
    CANDIDATE,
    STRONG_BASELINE,
    BenchConfig,
    same_caliber,
)
from vllm_evolve.bench.dispatch import RemoteBench, run_remote_bench_config
from vllm_evolve.bench.gpu_selection import (
    AUTO_GPUS,
    bind_auto_gpu_selection,
    is_auto_gpu_selection,
)
from vllm_evolve.bench.real_acceptance import (
    FORMAL_SEEDS,
    SCENARIOS,
    candidate_execution_ok,
    evaluate_suite,
)
from vllm_evolve.bench.real_acceptance import metric_summary as metric_summary
from vllm_evolve.engine.quality_remote import make_remote_quality_measure_fn
from vllm_evolve.engine.quality_runner import (
    is_quality_certified,
    measured_quality_verdict,
)

MIN_FORMAL_REQUESTS = 512


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fmt(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _report_runs(record: dict):
    for scenario in SCENARIOS:
        for role, run in (record.get("pairs", {}).get(scenario) or {}).items():
            yield scenario, role, run
        control = record.get("controls", {}).get(scenario)
        if control:
            yield scenario, "mechanism-control", control


def _pressure_rows(record: dict) -> list[dict]:
    rows = []
    for scenario, role, run in _report_runs(record):
        eval_result = dict(run.get("eval_result") or {})
        validity = dict(eval_result.get("workload_validity") or {})
        provenance = dict(eval_result.get("plugin_provenance") or {})
        for seed_result in validity.get("per_seed") or []:
            gpu_map = dict(seed_result.get("gpu") or {})
            selected = [
                str(value) for value in seed_result.get("selected_gpus") or []
            ]
            per_replica_vllm = dict(
                seed_result.get("vllm_per_replica") or {}
            )
            per_replica_client = dict(
                seed_result.get("client_inflight_per_replica") or {}
            )
            for gpu_index, gpu in gpu_map.items():
                replica_index = (
                    selected.index(str(gpu_index))
                    if str(gpu_index) in selected else 0
                )
                vllm = (
                    per_replica_vllm.get(str(replica_index))
                    or seed_result.get("vllm")
                    or {}
                )
                client = (
                    per_replica_client.get(str(replica_index))
                    or seed_result.get("client_inflight")
                    or {}
                )
                rows.append({
                    "scenario": scenario,
                    "role": role,
                    "seed": seed_result.get("seed"),
                    "valid": seed_result.get("valid"),
                    "verdict": seed_result.get("verdict"),
                    "gpu_index": gpu_index,
                    "gpu": gpu,
                    "vllm": vllm,
                    "client": client,
                    "measurement": seed_result.get("measurement") or {},
                    "plateau": seed_result.get("plateau") or {},
                    "provenance": provenance,
                })
    return rows


def render_report(record: dict) -> str:
    acceptance = record["acceptance"]
    lines = [
        "# vLLM-Evolve real-GPU verification",
        "",
        f"- Decision: **{'ACCEPT' if acceptance['accepted'] else 'REJECT'}**",
        f"- Model: `{record['model']}`",
        f"- Remote: `{record['remote']}`",
        f"- Visible GPUs: `{record['gpus']}` (hard limit: at most two)",
        f"- Candidate: `{record['policy_path']}`",
        f"- Mechanism control: `{record['control_path']}`",
        f"- Quality certified: `{acceptance['quality_ok']}`",
        f"- Median gain vs strong baseline: "
        f"`{_fmt(acceptance['median_gain_pct'])}%`",
        "",
        "## Three-scenario results",
        "",
        "| Scenario | Paired gain | Paired 95% CI | Gate | "
        "Req/s (base→cand) | Out tok/s (base→cand) | "
        "TTFT p95 ms (base→cand) | TPOT p95 ms (base→cand) | "
        "E2E p95 ms (base→cand) | Complete | Error rate | CV |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in acceptance["rows"]:
        base = row["baseline"]
        cand = row["candidate"]
        comparison = row["comparison"]
        lines.append(
            f"| {row['scenario']} | {_fmt(row['gain_pct'])}% | "
            f"[{_fmt(comparison.get('ci_low_pct'))}%, "
            f"{_fmt(comparison.get('ci_high_pct'))}%] | "
            f"{comparison.get('gate_ok')} | "
            f"{_fmt(base['request_throughput_req_s'])}→"
            f"{_fmt(cand['request_throughput_req_s'])} | "
            f"{_fmt(base['output_throughput_tok_s'])}→"
            f"{_fmt(cand['output_throughput_tok_s'])} | "
            f"{_fmt(base['ttft_ms']['p95'])}→{_fmt(cand['ttft_ms']['p95'])} | "
            f"{_fmt(base['tpot_ms']['p95'])}→{_fmt(cand['tpot_ms']['p95'])} | "
            f"{_fmt(base['e2e_ms']['p95'])}→{_fmt(cand['e2e_ms']['p95'])} | "
            f"{_fmt(cand['completion_rate'], 4)} | "
            f"{_fmt(cand['error_rate'], 4)} | "
            f"{_fmt(cand['primary_cv'], 4)} |"
        )
    mechanism = acceptance["mechanism_evidence"]
    lines += [
        "",
        "## Mechanism ablation",
        "",
        f"Mechanism: {mechanism['mechanism']}. "
        f"Supported: **{mechanism['supported']}**; median contribution: "
        f"`{_fmt(mechanism['median_gain_pct'])}%`.",
        "",
        "| Scenario | Gain over control | Execution valid | Completion valid |",
        "|---|---:|---:|---:|",
    ]
    for row in mechanism["rows"]:
        lines.append(
            f"| {row['scenario']} | {_fmt(row['gain_pct'])}% | "
            f"{row['execution_ok']} | {row['completion_ok']} |"
        )
    pressure_rows = _pressure_rows(record)
    lines += [
        "",
        "## Formal workload validity",
        "",
        "Every row below is measured-window evidence. A performance delta is "
        "nonqualifying unless every applicable row is valid.",
        "",
        "| Scenario / role / seed / GPU | Verdict | Util p10/p50/p95 | Duty | "
        "Memory p50/peak | Power p50 | SM clock p50 | KV p50/p95/peak | "
        "Running p50/p95/max | Waiting p50/p95/max | Scheduler Δ | "
        "Offered/actual QPS | Inflight p50/p95/max | Complete/errors | Plateau |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|:---:|",
    ]
    for row in pressure_rows:
        gpu = row["gpu"]
        vllm = row["vllm"]
        measurement = row["measurement"]
        client = row["client"]
        util = gpu.get("utilization_pct") or {}
        memory = gpu.get("memory_ratio") or {}
        power = gpu.get("power_w") or {}
        clock = gpu.get("sm_clock_mhz") or {}
        kv = vllm.get("kv_cache_occupancy") or {}
        running = vllm.get("running_requests") or {}
        waiting = vllm.get("waiting_requests") or {}
        completed = measurement.get("completed")
        errors = measurement.get("errors")
        lines.append(
            f"| {row['scenario']} / {row['role']} / s{row['seed']} / "
            f"GPU {row['gpu_index']} | {row['verdict']} | "
            f"{_fmt(util.get('p10'))}/{_fmt(util.get('p50'))}/"
            f"{_fmt(util.get('p95'))} | "
            f"{_fmt(gpu.get('active_duty_cycle'), 3)} | "
            f"{_fmt(memory.get('p50'), 3)}/{_fmt(memory.get('max'), 3)} | "
            f"{_fmt(power.get('p50'))} | {_fmt(clock.get('p50'))} | "
            f"{_fmt(kv.get('p50'), 3)}/{_fmt(kv.get('p95'), 3)}/"
            f"{_fmt(kv.get('max'), 3)} | "
            f"{_fmt(running.get('p50'))}/{_fmt(running.get('p95'))}/"
            f"{_fmt(running.get('max'))} | "
            f"{_fmt(waiting.get('p50'))}/{_fmt(waiting.get('p95'))}/"
            f"{_fmt(waiting.get('max'))} | "
            f"{_fmt(vllm.get('scheduler_invocations_delta'))} | "
            f"{_fmt(measurement.get('offered_qps'))}/"
            f"{_fmt(measurement.get('actual_sent_qps'))} | "
            f"{_fmt(client.get('p50'))}/{_fmt(client.get('p95'))}/"
            f"{_fmt(client.get('max'))} | "
            f"{_fmt(completed, 0)}/{_fmt(errors, 0)} | "
            f"{row['plateau'].get('valid')} |"
        )
    lines += [
        "",
        "## Mechanism action counters",
        "",
        "Counters are cumulative for the complete multi-seed server run; replica "
        "groups sum the two independently nonce-verified plugin logs.",
        "",
        "| Scenario / role | Effective | Applicable | Invocations | Reorders | "
        "Deferred | Forced admits | Preemptions | Fallbacks |",
        "|---|:---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, role, run in _report_runs(record):
        eval_result = dict(run.get("eval_result") or {})
        provenance = dict(eval_result.get("plugin_provenance") or {})
        lines.append(
            f"| {scenario} / {role} | {eval_result.get('effective')} | "
            f"{eval_result.get('mechanism_applicable')} | "
            f"{_fmt(provenance.get('policy_invocation_count'), 0)} | "
            f"{_fmt(provenance.get('reorder_action_count'), 0)} | "
            f"{_fmt(provenance.get('deferred_request_actions'), 0)} | "
            f"{_fmt(provenance.get('starvation_forced_admissions'), 0)} | "
            f"{_fmt(provenance.get('preemption_count'), 0)} | "
            f"{_fmt(provenance.get('fallback_count'), 0)} |"
        )
    lines += [
        "",
        "## GPU and artifact evidence",
        "",
    ]
    for scenario, pair in record["pairs"].items():
        for role, run in pair.items():
            status = run.get("status") or {}
            summary = status.get("gpu_summary") or {}
            peaks = ", ".join(
                f"GPU {idx}: {gpu.get('max_memory_used_mib')} MiB, "
                f"{gpu.get('max_utilization_gpu_pct')}%"
                for idx, gpu in (summary.get("per_gpu") or {}).items()
            ) or "n/a"
            lines.append(
                f"- `{scenario}/{role}`: `{run.get('local_run_dir')}`; {peaks}"
            )
        control = record.get("controls", {}).get(scenario)
        if control:
            status = control.get("status") or {}
            summary = status.get("gpu_summary") or {}
            peaks = ", ".join(
                f"GPU {idx}: {gpu.get('max_memory_used_mib')} MiB, "
                f"{gpu.get('max_utilization_gpu_pct')}%"
                for idx, gpu in (summary.get("per_gpu") or {}).items()
            ) or "n/a"
            lines.append(
                f"- `{scenario}/mechanism-control`: "
                f"`{control.get('local_run_dir')}`; {peaks}"
            )
    for evidence_dir in record.get("quality", {}).get("evidence_dirs", []):
        lines.append(f"- `quality`: `{evidence_dir}`")
    lines += [
        "",
        "The JSON beside this report is authoritative. Simulator scores are not "
        "reported as real GPU gains; a candidate is accepted only when the real-vLLM "
        "execution, stability, quality, and mechanism gates all pass.",
        "",
    ]
    return "\n".join(lines)


def _canonical_sha256(payload: dict) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_frozen_bench_config(path: str | Path) -> tuple[BenchConfig, dict]:
    """Load and validate the calibration-locked formal execution template."""
    frozen_path = Path(path).resolve()
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    # Accept both a direct bench_config.json and an immutable wrapper carrying
    # the config plus calibration metadata.
    config_payload = payload.get("bench_config") if "bench_config" in payload else payload
    if not isinstance(config_payload, dict):
        raise ValueError("frozen bench config must be a JSON object")
    config = BenchConfig.from_dict(config_payload)
    validity = dict(config.workload.workload_spec.get("validity") or {})
    if validity.get("required") is not True:
        raise ValueError(
            "frozen bench config must require valid_saturated_real_vllm"
        )
    plateau = dict(validity.get("plateau_reference") or {})
    required_plateau = {
        "base_offered_qps",
        "higher_offered_qps",
        "base_achieved_throughput",
        "higher_achieved_throughput",
    }
    missing_plateau = sorted(required_plateau - set(plateau))
    if missing_plateau:
        raise ValueError(
            "frozen bench config is missing explicit plateau fields: "
            f"{missing_plateau}"
        )
    if not is_auto_gpu_selection(config.environment.gpus):
        selected_gpus = tuple(
            item.strip()
            for item in str(config.environment.gpus).split(",")
            if item.strip()
        )
        if len(selected_gpus) != 2 or len(set(selected_gpus)) != 2:
            raise ValueError(
                "formal real-vLLM suite requires exactly two distinct selected GPUs"
            )
    parallel_mode = str(
        config.workload.workload_spec.get("parallel_mode")
        or (
            "tp2"
            if config.engine.tensor_parallel_size == 2 else ""
        )
    )
    if parallel_mode == "tp2" and config.engine.tensor_parallel_size != 2:
        raise ValueError("tp2 formal mode requires tensor_parallel_size=2")
    if (
        parallel_mode == "dual_replica"
        and config.engine.tensor_parallel_size != 1
    ):
        raise ValueError(
            "dual_replica formal mode requires tensor_parallel_size=1"
        )
    if parallel_mode not in {"tp2", "dual_replica"}:
        raise ValueError(
            "formal config must select tp2 or dual_replica parallel_mode"
        )
    config.workload.workload_spec["parallel_mode"] = parallel_mode
    if tuple(config.statistical.seed_tiers) != FORMAL_SEEDS:
        raise ValueError(
            f"formal seed tiers must be exactly {list(FORMAL_SEEDS)}"
        )
    if config.statistical.primary_metric not in {
        "goodput_req_s",
        "request_throughput_req_s",
        "output_throughput_tok_s",
    }:
        raise ValueError(
            "formal evolution primary metric must be a higher-is-better "
            "throughput/goodput metric"
        )
    return config, {
        "path": str(frozen_path),
        "file_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "bench_config_sha256": _canonical_sha256(config.to_dict()),
        "calibration_metadata": (
            payload.get("calibration") if "bench_config" in payload else None
        ),
    }


def _formal_workload_fields(workload: dict, scenario: str) -> dict:
    if workload.get("source_kind") != "official_burstgpt":
        raise ValueError(
            f"{scenario} is not an official BurstGPT materialization"
        )
    if workload.get("scenario") != scenario:
        raise ValueError(
            f"workload scenario mismatch: expected {scenario!r}, "
            f"got {workload.get('scenario')!r}"
        )
    replay = dict(workload.get("replay") or {})
    measured_requests = int(replay.get("measured_requests") or 0)
    duration_s = float(replay.get("nominal_arrival_span_s") or 0.0)
    target_qps = float(replay.get("target_qps") or 0.0)
    if measured_requests < MIN_FORMAL_REQUESTS:
        raise ValueError(
            f"{scenario} has {measured_requests} requests; "
            f"formal minimum is {MIN_FORMAL_REQUESTS}"
        )
    # Arrival span is not the measured window.  A deliberately overloaded
    # replay can submit its finite trace quickly and then spend >120 seconds
    # draining the live vLLM queue.  Requiring 120 seconds of *arrivals* at
    # hundreds of QPS would manufacture tens of thousands of requests and turn
    # every seed into a multi-hour drain.  The immutable post-run validity
    # evidence remains responsible for enforcing the 120-second hard gate on
    # the actual measured window.
    if duration_s <= 0:
        raise ValueError(
            f"{scenario} nominal arrival span must be positive"
        )
    if target_qps <= 0:
        raise ValueError(f"{scenario} target_qps must be positive")
    if replay.get("arrival_scaling_only") is not True:
        raise ValueError(
            f"{scenario} must preserve tokens and scale arrivals only"
        )
    return {
        "measured_requests": measured_requests,
        "nominal_arrival_span_s": duration_s,
        "target_qps": target_qps,
    }


def _config(
    *,
    frozen: BenchConfig,
    runner_kind: str,
    policy_path: str | None,
    workload_path: Path,
    workload: dict,
    scenario: str,
    port: int,
    max_seeds: int,
    artifact_root: Path,
) -> BenchConfig:
    formal = _formal_workload_fields(workload, scenario)
    config = BenchConfig.from_dict(frozen.to_dict())
    config.runner.runner_kind = runner_kind
    config.runner.policy_path = policy_path
    config.runner.port = int(port)
    config.runner.local_artifact_root = str(artifact_root)
    config.engine.scheduler_cls = (
            "generated_scheduler.EvolvedScheduler"
            if runner_kind == CANDIDATE else None
    )
    config.statistical.max_seeds = int(max_seeds)
    config.workload.trace_path = str(workload_path)
    config.workload.load_mode = "open"
    config.workload.concurrency = formal["measured_requests"]
    config.workload.n_requests = formal["measured_requests"]
    config.workload.arrival_rate_qps = formal["target_qps"]
    workload_spec = dict(config.workload.workload_spec)
    validity = dict(workload_spec.get("validity") or {})
    validity["required"] = True
    validity["offered_qps"] = formal["target_qps"]
    workload_spec.update({
        "source": "real_vllm",
        "source_kind": "official_burstgpt",
        "scenario": scenario,
        "workload_payload_sha256": workload.get("payload_sha256"),
        "official_release": workload.get("official_release"),
        "split": workload.get("split"),
        "arrival_scaling_only": True,
        "measured_requests": formal["measured_requests"],
        "nominal_arrival_span_s": formal["nominal_arrival_span_s"],
        "validity": validity,
    })
    config.workload.workload_spec = workload_spec
    return config


def _run_and_record(
    config: BenchConfig,
    record: dict,
    key: str,
    progress: Path,
    *,
    git_sha: str,
) -> RemoteBench:
    result = run_remote_bench_config(config, git_sha=git_sha)
    record[key] = result.to_dict()
    _write(progress, record)
    return result


def _bind_formal_gpu_selection(config: BenchConfig, requested: str) -> None:
    """Bind one GPU group while preserving an earlier automatic freeze.

    A winner-search eval result may already carry the exact auto-selected IDs,
    UUIDs, and inventory snapshot.  ``--gpus auto`` must reuse that group for
    held-out A/B/C rather than silently treating the formal suite as a new
    selection round.  A config without prior evidence is discovered normally.
    """
    evidence = dict(config.environment.gpu_selection_evidence or {})
    already_frozen_auto = (
        requested == AUTO_GPUS
        and config.environment.gpu_selection_mode == AUTO_GPUS
        and evidence.get("mode") == AUTO_GPUS
        and evidence.get("resolved") == config.environment.gpus
    )
    if requested != "config" and not already_frozen_auto:
        config.environment.gpus = str(requested)
        config.environment.gpu_selection_mode = "fixed"
        config.environment.gpu_selection_evidence = {}
    bind_auto_gpu_selection(config)


def run_suite(
    *,
    policy_path: str | Path,
    workloads_dir: str | Path,
    out_dir: str | Path,
    frozen_bench_config: str | Path,
    control_path: str | Path | None = None,
    mechanism_name: str = "candidate mechanism vs supplied control",
    base_port: int = 8260,
    gpus: str = AUTO_GPUS,
    required_action_counters: list[str] | None = None,
) -> dict:
    if control_path is None:
        raise ValueError(
            "a real mechanism-control policy is required for adoption evidence"
        )
    required_action_counters = list(dict.fromkeys(required_action_counters or []))
    if not required_action_counters:
        raise ValueError(
            "at least one --required-action-counter declared by the mechanism is required"
        )
    policy_path = Path(policy_path).resolve()
    control_path = Path(control_path).resolve()
    workloads_dir = Path(workloads_dir).resolve()
    frozen, frozen_provenance = load_frozen_bench_config(frozen_bench_config)
    _bind_formal_gpu_selection(frozen, gpus)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    artifact_root = out / "artifacts"
    progress = out / "progress.json"
    suite_git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    record: dict = {
        "schema_version": 1,
        "source": "real_vllm",
        "outcome_class": "real_acceptance_suite",
        "policy_path": str(policy_path),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "control_path": str(control_path),
        "control_sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
        "mechanism_name": mechanism_name,
        "required_action_counters": required_action_counters,
        "model": frozen.engine.model,
        "gpus": frozen.environment.gpus,
        "gpu_selection": dict(frozen.environment.gpu_selection_evidence or {}),
        "remote": frozen.runner.remote,
        "parallel_mode": (
            frozen.workload.workload_spec["parallel_mode"]
        ),
        "frozen_bench_config": frozen_provenance,
        "git_sha": suite_git_sha,
        "smoke": None,
        "pairs": {},
        "controls": {},
        "quality": None,
        "acceptance": None,
    }
    _write(progress, record)

    loaded = {
        scenario: (
            workloads_dir / f"{scenario}.json",
            json.loads((workloads_dir / f"{scenario}.json").read_text(encoding="utf-8")),
        )
        for scenario in SCENARIOS
    }
    port = base_port
    smoke_path, smoke_workload = loaded["burstgpt_saturated"]
    smoke_config = _config(
        frozen=frozen,
        runner_kind=CANDIDATE,
        policy_path=str(policy_path),
        workload_path=smoke_path,
        workload=smoke_workload,
        scenario="burstgpt_saturated",
        port=port,
        max_seeds=1,
        artifact_root=artifact_root,
    )
    smoke = _run_and_record(
        smoke_config,
        record,
        "smoke",
        progress,
        git_sha=suite_git_sha,
    )
    smoke_eval = smoke.eval_result
    if not candidate_execution_ok(smoke_eval):
        raise RuntimeError(
            "candidate smoke did not execute validly: "
            f"{smoke.local_run_dir}"
        )

    pairs = {}
    controls = {}
    for scenario in SCENARIOS:
        path, workload = loaded[scenario]
        baseline_config = _config(
            frozen=frozen,
            runner_kind=STRONG_BASELINE,
            policy_path=None,
            workload_path=path,
            workload=workload,
            scenario=scenario,
            port=port + 1,
            max_seeds=3,
            artifact_root=artifact_root,
        )
        candidate_config = _config(
            frozen=frozen,
            runner_kind=CANDIDATE,
            policy_path=str(policy_path),
            workload_path=path,
            workload=workload,
            scenario=scenario,
            port=port + 2,
            max_seeds=3,
            artifact_root=artifact_root,
        )
        caliber_ok, caliber_diffs = same_caliber(baseline_config, candidate_config)
        if not caliber_ok:
            raise RuntimeError(f"same-caliber violation for {scenario}: {caliber_diffs}")
        baseline = run_remote_bench_config(
            baseline_config, git_sha=suite_git_sha
        )
        record["pairs"].setdefault(scenario, {})["baseline"] = baseline.to_dict()
        _write(progress, record)
        candidate = run_remote_bench_config(
            candidate_config, git_sha=suite_git_sha
        )
        record["pairs"][scenario]["candidate"] = candidate.to_dict()
        _write(progress, record)
        pairs[scenario] = {
            "baseline": baseline.eval_result,
            "candidate": candidate.eval_result,
        }
        control_config = _config(
            frozen=frozen,
            runner_kind=CANDIDATE,
            policy_path=str(control_path),
            workload_path=path,
            workload=workload,
            scenario=scenario,
            port=port + 3,
            max_seeds=3,
            artifact_root=artifact_root,
        )
        control = run_remote_bench_config(
            control_config, git_sha=suite_git_sha
        )
        record["controls"][scenario] = control.to_dict()
        controls[scenario] = control.eval_result
        _write(progress, record)
        port += 4

    quality_base = {
        **frozen.to_dict()["engine"],
        "gpus": frozen.environment.gpus,
        "remote": frozen.runner.remote,
        "profile": frozen.workload.regime,
        "remote_repo": frozen.runner.remote_repo,
        "conda_env": frozen.runner.conda_env,
        "conda_sh": frozen.runner.conda_sh,
        "remote_workspace": frozen.runner.remote_workspace,
        "local_artifact_root": str(artifact_root),
        "hf_endpoint": frozen.runner.hf_endpoint,
    }
    quality_factory = make_remote_quality_measure_fn(quality_base)
    quality_verdict = measured_quality_verdict(
        quality_factory(quality_base),
        quality_factory(quality_base),
    )
    record["quality"] = (
        quality_verdict.to_dict() if quality_verdict is not None else {
            "ok": False,
            "reasons": ["remote quality measurement unavailable"],
            "measured": {},
        }
    )
    record["quality"]["evidence_dirs"] = list(
        getattr(quality_factory, "evidence_dirs", [])
    )
    acceptance = evaluate_suite(
        pairs,
        quality_ok=is_quality_certified(quality_verdict),
        controls=controls,
        require_mechanism_control=True,
        mechanism_name=mechanism_name,
        n_boot=frozen.statistical.n_boot,
    )
    action_rows = []
    for scenario in SCENARIOS:
        candidate_eval = pairs[scenario]["candidate"]
        provenance = candidate_eval.get("plugin_provenance") or {}
        applicable = candidate_eval.get("mechanism_applicable") is not False
        fired = {
            counter: provenance.get(counter)
            for counter in required_action_counters
        }
        action_rows.append({
            "scenario": scenario,
            "mechanism_applicable": applicable,
            "counters": fired,
            "all_declared_counters_fired": (
                applicable
                and all(isinstance(value, (int, float)) and value > 0 for value in fired.values())
            ),
        })
    action_counters_ok = sum(
        row["all_declared_counters_fired"] for row in action_rows
    ) >= 2
    acceptance["required_action_counters"] = required_action_counters
    acceptance["action_counter_rows"] = action_rows
    acceptance["action_counters_ok"] = action_counters_ok
    acceptance["accepted"] = acceptance["accepted"] and action_counters_ok
    record["acceptance"] = acceptance
    from vllm_evolve.bench.eval_result import (
        acceptance_evidence_sha256,
        build_acceptance_artifact_manifest,
    )

    eval_result_paths = [
        str(run.get("local_eval_path"))
        for scenario in SCENARIOS
        for run in (
            record["pairs"][scenario]["baseline"],
            record["pairs"][scenario]["candidate"],
            record["controls"][scenario],
        )
    ]
    record["artifact_manifest"] = build_acceptance_artifact_manifest(
        artifact_root,
        eval_result_paths=eval_result_paths,
        quality_evidence_dirs=list(record["quality"].get("evidence_dirs") or []),
    )
    record["evidence_sha256"] = acceptance_evidence_sha256(record)
    _write(progress, record)
    _write(out / "result.json", record)
    (out / "report.md").write_text(render_report(record), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm_evolve.engine.real_evolve_suite")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument(
        "--mechanism-name",
        default="candidate mechanism vs supplied control",
    )
    parser.add_argument(
        "--required-action-counter",
        action="append",
        required=True,
        dest="required_action_counters",
        help="Plugin provenance counter declared by the candidate mechanism; repeat as needed.",
    )
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--frozen-bench-config",
        required=True,
        help=(
            "Calibration-locked BenchConfig JSON. Engine/environment/statistical "
            "knobs are cloned exactly for baseline, candidate, and control."
        ),
    )
    parser.add_argument("--base-port", type=int, default=8260)
    parser.add_argument(
        "--gpus",
        default=AUTO_GPUS,
        help=(
            "GPU selection: auto (default) discovers and freezes one idle "
            "same-caliber pair for the full suite; use config to honor the "
            "frozen BenchConfig value, or pass two explicit CUDA device IDs."
        ),
    )
    args = parser.parse_args(argv)
    result = run_suite(
        policy_path=args.policy,
        control_path=args.control,
        workloads_dir=args.workloads,
        out_dir=args.out,
        frozen_bench_config=args.frozen_bench_config,
        mechanism_name=args.mechanism_name,
        base_port=args.base_port,
        gpus=args.gpus,
        required_action_counters=args.required_action_counters,
    )
    print(json.dumps(result["acceptance"], indent=2))
    return 0 if result["acceptance"]["accepted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
