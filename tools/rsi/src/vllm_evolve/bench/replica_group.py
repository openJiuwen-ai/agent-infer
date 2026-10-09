"""Two-single-GPU replica execution and evidence aggregation.

One logical formal benchmark may be served by two identical TP=1 replicas.
Each replica receives a deterministic disjoint partition of the same official
BurstGPT replay, starts every seed through a shared barrier, and retains its
own raw server/GPU/queue evidence. Fitness is computed only from the combined
request records after both replicas independently pass their per-GPU and
per-scheduler pressure gates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import secrets
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from vllm_evolve.bench.config import BenchConfig
from vllm_evolve.bench.datasets.burstgpt import paired_mass_balance_owners
from vllm_evolve.bench.eval_result import validate_eval_result
from vllm_evolve.bench.metrics import BenchMetrics, RequestRecord
from vllm_evolve.bench.profiles import primary_metric_fn
from vllm_evolve.bench.runner import aggregate
from vllm_evolve.bench.slo import SLO, evaluate_slo
from vllm_evolve.bench.validity_artifacts import (
    _actual_sent_qps,
    _inside,
    _jsonl,
    summarize_client_inflight,
)
from vllm_evolve.bench.workload_validity import (
    CLIENT_UNDERPOWERED,
    EXTERNAL_CONTAMINATION,
    GPU_IMBALANCE,
    LOW_KV_OCCUPANCY,
    NO_QUEUE_PRESSURE,
    UNDERLOADED_GPU,
    UNSTABLE_OR_INCOMPLETE,
    VALID,
    ValidityThresholds,
    evaluate_workload_validity,
)

DUAL_REPLICA = "dual_replica"
REPLICA_MEMBER = "replica_member"
_REPLICA_COUNT = 2
_MAX_START_SKEW_S = 1.0


def _canonical_sha(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def split_scaled_replay_payload(
    payload: dict,
    *,
    replica_count: int = _REPLICA_COUNT,
) -> list[dict]:
    """Partition replay occurrences with cycle-rotated paired mass balance.

    Every adjacent arrival pair is split across the replicas.  Pair orientation
    minimizes cumulative prompt/output token-mass skew, preventing complementary
    FCFS streams from entering long heavy/light phases.  Orientation swaps on
    the next cycle, so both members still receive every official request shape
    equally often over each cycle pair.  Every expanded occurrence is assigned
    exactly once and keeps its original arrival and token shape.
    """
    if replica_count != _REPLICA_COUNT:
        raise ValueError("formal replica mode requires exactly two replicas")
    if (
        payload.get("source_kind") != "official_burstgpt"
        or int(payload.get("schema_version") or 0) < 2
    ):
        raise ValueError("dual replica mode requires official BurstGPT schema v2")
    requests = list(payload.get("requests") or [])
    replay = dict(payload.get("replay") or {})
    cycles = int(replay.get("cycles") or 0)
    total_measured = int(replay.get("measured_requests") or 0)
    total_qps = float(replay.get("target_qps") or 0.0)
    if len(requests) < replica_count or cycles < 1 or total_measured < 1:
        raise ValueError("scaled BurstGPT payload is incomplete")
    if total_measured != len(requests) * cycles:
        raise ValueError("replay measured_requests does not match base_requests*cycles")
    if len(requests) % replica_count or cycles % replica_count:
        raise ValueError(
            "cycle-rotated replica stripes require request and cycle counts "
            "divisible by replica_count"
        )

    base_prompt_tokens = sum(
        int(request.get("num_prompt_tokens") or 0) for request in requests
    )
    base_output_tokens = sum(
        int(request.get("num_output_tokens") or 0) for request in requests
    )
    base_owners = paired_mass_balance_owners(requests)
    members = []
    for index in range(replica_count):
        source_cycles = list(range(cycles))
        member_measured = total_measured // replica_count
        member_replay = {
            **replay,
            "target_qps": total_qps * member_measured / total_measured,
            "base_requests": len(requests) // replica_count,
            "measured_requests": member_measured,
            "aggregate_target_qps": total_qps,
            "aggregate_measured_requests": total_measured,
        }
        core = {
            key: copy.deepcopy(value)
            for key, value in payload.items()
            if key not in {"payload_sha256", "requests", "replay"}
        }
        core.update({
            "requests": copy.deepcopy(requests),
            "replay": member_replay,
            "replica_partition": {
                "strategy": "cycle_rotated_paired_mass_balance",
                "index": index,
                "count": replica_count,
                "source_payload_sha256": payload.get("payload_sha256"),
                "source_split": payload.get("split"),
                "source_cycle_indices": source_cycles,
                "source_cycle_period_s": float(
                    replay.get("cycle_period_s") or 0.0
                ),
                "source_base_requests": len(requests),
                "base_owner_by_request_index": base_owners,
                "base_prompt_tokens": base_prompt_tokens,
                "base_output_tokens": base_output_tokens,
                "member_prompt_tokens": (
                    base_prompt_tokens * cycles // replica_count
                ),
                "member_output_tokens": (
                    base_output_tokens * cycles // replica_count
                ),
                "tokens_unchanged": True,
                "arrival_timestamps_unchanged": True,
                "member_source_order_unchanged": True,
            },
        })
        members.append({**core, "payload_sha256": _canonical_sha(core)})
    if sum(
        member["replay"]["measured_requests"] for member in members
    ) != total_measured:
        raise AssertionError("replica partition lost measured requests")
    for member in members:
        if member["requests"] != requests:
            raise AssertionError(
                "replica partition changed the official base fragment"
            )
    assigned_occurrences = sorted(
        (cycle, request_index)
        for member in members
        for cycle in member["replica_partition"]["source_cycle_indices"]
        for request_index in range(len(requests))
        if (
            (base_owners[request_index] + cycle) % replica_count
            == member["replica_partition"]["index"]
        )
    )
    expected_occurrences = [
        (cycle, request_index)
        for cycle in range(cycles)
        for request_index in range(len(requests))
    ]
    if assigned_occurrences != expected_occurrences:
        raise AssertionError(
            "replica partition lost or duplicated a replay occurrence"
        )
    return members


def _raw_record(row: dict, *, origin_s: float) -> RequestRecord:
    submit = float(row.get("submit_time_s") or 0.0)
    first = row.get("first_token_time_s")
    end = row.get("end_time_s")
    first_value = float(first) if first is not None else submit
    end_value = float(end) if end is not None else first_value
    output_tokens = int(row.get("num_output_tokens") or 0)
    return RequestRecord(
        request_id=str(row.get("request_id") or ""),
        arrival_time_s=submit - origin_s,
        ttft_ms=max(0.0, first_value - submit) * 1000.0,
        tpot_ms=(
            max(0.0, end_value - first_value) * 1000.0
            / max(1, output_tokens - 1)
            if output_tokens >= 2 else 0.0
        ),
        e2e_ms=max(0.0, end_value - submit) * 1000.0,
        num_prompt_tokens=int(row.get("num_prompt_tokens") or 0),
        num_output_tokens=output_tokens,
        success=bool(row.get("success")),
        error=row.get("error"),
    )


def _member_artifacts(remote_bench) -> dict:
    root = Path(remote_bench.local_run_dir)
    windows_payload = json.loads(
        (root / "measurement_windows.json").read_text(encoding="utf-8")
    )
    return {
        "root": root,
        "windows": {
            int(window["seed"]): window
            for window in windows_payload.get("windows") or []
        },
        "gpu": _jsonl(root / "gpu_samples.jsonl"),
        "vllm": _jsonl(root / "vllm_metrics.jsonl"),
        "raw": _jsonl(root / "raw_requests.jsonl"),
    }


def _verdict_from_categories(categories: list[str]) -> str:
    for category in (
        EXTERNAL_CONTAMINATION,
        CLIENT_UNDERPOWERED,
        UNSTABLE_OR_INCOMPLETE,
        GPU_IMBALANCE,
        LOW_KV_OCCUPANCY,
        NO_QUEUE_PRESSURE,
        UNDERLOADED_GPU,
    ):
        if category in categories:
            return category
    return VALID


def _group_validity(
    config: BenchConfig,
    members: list,
    member_configs: list[BenchConfig],
    artifacts: list[dict],
    seeds: list[int],
) -> dict:
    protocol = dict(config.workload.workload_spec.get("validity") or {})
    thresholds = ValidityThresholds()
    member_thresholds = replace(
        thresholds,
        selected_gpu_count=1,
        min_measured_requests=1,
    )
    results = []
    for seed in seeds:
        replica_results = []
        starts = []
        ends = []
        aggregate_raw = []
        for index, (remote, member_config, evidence) in enumerate(
            zip(members, member_configs, artifacts)
        ):
            window = evidence["windows"].get(seed)
            if window is None:
                replica_results.append({
                    "replica_index": index,
                    "valid": False,
                    "verdict": UNSTABLE_OR_INCOMPLETE,
                    "reasons": ["measurement window is missing"],
                })
                continue
            starts.append(float(window["started_at_unix_s"]))
            ends.append(float(window["ended_at_unix_s"]))
            seed_raw = [
                row for row in evidence["raw"]
                if int(row.get("seed", -9999)) == seed
            ]
            aggregate_raw.extend(seed_raw)
            metric_row = next(
                (
                    row
                    for row in remote.eval_result.get("raw_per_seed_metrics") or []
                    if int(row.get("seed", -9999)) == seed
                ),
                {},
            )
            metrics = dict(metric_row.get("metrics") or {})
            member_protocol = dict(
                member_config.workload.workload_spec.get("validity") or {}
            )
            duration = float(window.get("duration_s") or 0.0)
            member_result = evaluate_workload_validity(
                gpu_samples=[
                    row for row in evidence["gpu"] if _inside(row, window)
                ],
                vllm_samples=[
                    row
                    for row in evidence["vllm"]
                    if int(row.get("seed", -9999)) == seed and _inside(row, window)
                ],
                selected_gpus=(member_config.environment.gpus,),
                max_num_seqs=member_config.engine.max_num_seqs,
                measured_duration_s=duration,
                measured_requests=int(
                    window.get("measured_requests") or len(seed_raw)
                ),
                completed_requests=int(metrics.get("num_completed") or 0),
                error_count=int(metrics.get("num_failed") or 0),
                offered_qps=float(member_protocol.get("offered_qps") or 0.0),
                actual_sent_qps=_actual_sent_qps(seed_raw, duration),
                plateau_reference=protocol.get("plateau_reference"),
                contaminated=bool(
                    (remote.status or {}).get("gpu_summary", {}).get("contaminated")
                ),
                thresholds=member_thresholds,
            )
            member_result["client_inflight"] = summarize_client_inflight(seed_raw)
            member_result["replica_index"] = index
            member_result["run_dir"] = remote.local_run_dir
            replica_results.append(member_result)

        categories = []
        reasons = []
        for result in replica_results:
            categories.extend(result.get("failed_categories") or [])
            reasons.extend(
                f"replica {result.get('replica_index')}: {reason}"
                for reason in result.get("reasons") or []
            )
            if not result.get("valid") and not result.get("failed_categories"):
                categories.append(
                    str(result.get("verdict") or UNSTABLE_OR_INCOMPLETE)
                )
        start_skew = max(starts) - min(starts) if len(starts) == _REPLICA_COUNT else None
        overlap_duration = (
            min(ends) - max(starts)
            if len(starts) == _REPLICA_COUNT and len(ends) == _REPLICA_COUNT
            else 0.0
        )
        if start_skew is None or start_skew > _MAX_START_SKEW_S:
            categories.append(UNSTABLE_OR_INCOMPLETE)
            reasons.append(
                f"replica measured-window start skew {start_skew!r}s "
                f"exceeds {_MAX_START_SKEW_S}s"
            )
        measured_requests = len(aggregate_raw)
        completed = sum(bool(row.get("success")) for row in aggregate_raw)
        errors = measured_requests - completed
        if overlap_duration < thresholds.min_duration_s:
            categories.append(UNSTABLE_OR_INCOMPLETE)
            reasons.append(
                f"common measured duration {overlap_duration:.3f}s "
                f"< {thresholds.min_duration_s:.1f}s"
            )
        if measured_requests < thresholds.min_measured_requests:
            categories.append(UNSTABLE_OR_INCOMPLETE)
            reasons.append(
                f"aggregate measured requests {measured_requests} "
                f"< {thresholds.min_measured_requests}"
            )
        if errors or completed != measured_requests:
            categories.append(UNSTABLE_OR_INCOMPLETE)
            reasons.append(
                f"aggregate completion={completed}/{measured_requests}, errors={errors}"
            )

        per_gpu = {}
        for result in replica_results:
            per_gpu.update(result.get("gpu") or {})
        names = {gpu.get("name") for gpu in per_gpu.values()}
        capacities = {gpu.get("memory_total_mib") for gpu in per_gpu.values()}
        if (
            len(per_gpu) != _REPLICA_COUNT
            or len(names) != 1
            or None in names
            or len(capacities) != 1
            or None in capacities
        ):
            categories.append(UNSTABLE_OR_INCOMPLETE)
            reasons.append("replicas did not use two same-model, same-capacity GPUs")
        util_medians = [
            float(gpu["utilization_pct"]["p50"])
            for gpu in per_gpu.values()
            if gpu.get("utilization_pct", {}).get("p50") is not None
        ]
        if (
            len(util_medians) == _REPLICA_COUNT
            and max(util_medians) - min(util_medians)
            > thresholds.max_gpu_median_imbalance_points
        ):
            categories.append(GPU_IMBALANCE)
            reasons.append("replica GPU median utilization differs by over 20 points")

        verdict = _verdict_from_categories(list(dict.fromkeys(categories)))
        actual_qps = sum(
            float(result.get("measurement", {}).get("actual_sent_qps") or 0.0)
            for result in replica_results
        )
        results.append({
            "seed": seed,
            "source": "real_vllm",
            "parallel_mode": DUAL_REPLICA,
            "valid": verdict == VALID,
            "verdict": verdict,
            "reasons": reasons,
            "failed_categories": list(dict.fromkeys(categories)),
            "thresholds": {
                **thresholds.__dict__,
                "max_replica_start_skew_s": _MAX_START_SKEW_S,
            },
            "selected_gpus": [
                member.environment.gpus for member in member_configs
            ],
            "gpu": per_gpu,
            "vllm_per_replica": {
                str(result.get("replica_index")): result.get("vllm")
                for result in replica_results
            },
            "client_inflight_per_replica": {
                str(result.get("replica_index")): result.get("client_inflight")
                for result in replica_results
            },
            "measurement": {
                "common_duration_s": overlap_duration,
                "start_skew_s": start_skew,
                "requests": measured_requests,
                "completed": completed,
                "errors": errors,
                "completion_rate": (
                    completed / measured_requests if measured_requests else 0.0
                ),
                "offered_qps": float(protocol.get("offered_qps") or 0.0),
                "actual_sent_qps": actual_qps,
            },
            "plateau": (
                replica_results[0].get("plateau") if replica_results else None
            ),
            "replicas": replica_results,
        })
    first_invalid = next((row for row in results if not row["valid"]), None)
    return {
        "schema_version": 2,
        "required": True,
        "source": "real_vllm",
        "parallel_mode": DUAL_REPLICA,
        "valid": bool(results) and first_invalid is None,
        "verdict": (
            VALID
            if results and first_invalid is None
            else (
                first_invalid["verdict"]
                if first_invalid else UNSTABLE_OR_INCOMPLETE
            )
        ),
        "per_seed": results,
    }


def _combined_plugin_provenance(member_evals: list[dict]) -> tuple[dict | None, dict]:
    provenances = [
        dict(result.get("plugin_provenance") or {}) for result in member_evals
    ]
    if not any(provenances):
        return None, {
            "marker_verified": False,
            "effective": False,
            "mechanism_applicable": None,
        }
    marker_verified = all(result.get("marker_verified") is True for result in member_evals)
    applicable = [
        result.get("mechanism_applicable") for result in member_evals
    ]
    mechanism_applicable = (
        True if any(value is True for value in applicable)
        else False if all(value is False for value in applicable)
        else None
    )
    valid_member_execution = all(
        result.get("effective") is True
        or result.get("mechanism_applicable") is False
        for result in member_evals
    )
    any_effective = any(result.get("effective") is True for result in member_evals)
    combined = {
        "invoked": all(value.get("invoked") is True for value in provenances),
        "fallback": any(value.get("fallback") is True for value in provenances),
        "effective": valid_member_execution and any_effective,
        "mechanism_applicable": mechanism_applicable,
        "execution_valid": marker_verified and valid_member_execution,
        "replicas": provenances,
    }
    numeric_keys = {
        key
        for value in provenances
        for key, item in value.items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }
    for key in numeric_keys:
        combined[key] = sum(float(value.get(key) or 0.0) for value in provenances)
    return combined, {
        "marker_verified": marker_verified,
        "effective": combined["effective"],
        "mechanism_applicable": mechanism_applicable,
    }


def _combine_eval_result(
    config: BenchConfig,
    members: list,
    member_configs: list[BenchConfig],
    member_payloads: list[dict],
    source_payload: dict,
    group_dir: Path,
    started_at: float,
) -> dict:
    artifacts = [_member_artifacts(member) for member in members]
    seed_sets = [
        {
            int(row["seed"])
            for row in member.eval_result.get("raw_per_seed_metrics") or []
        }
        for member in members
    ]
    if not seed_sets or any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise RuntimeError(f"replica seed mismatch: {seed_sets}")
    seeds = sorted(seed_sets[0])
    raw_per_seed = []
    primary_fn = primary_metric_fn(config.statistical.primary_metric)
    slo = SLO(
        ttft_ms=config.statistical.slo.get("ttft_ms"),
        tpot_ms=config.statistical.slo.get("tpot_ms"),
        e2e_ms=config.statistical.slo.get("e2e_ms"),
    )
    for seed in seeds:
        windows = [evidence["windows"][seed] for evidence in artifacts]
        origin = min(float(window["started_at_unix_s"]) for window in windows)
        end = max(float(window["ended_at_unix_s"]) for window in windows)
        rows = [
            row
            for evidence in artifacts
            for row in evidence["raw"]
            if int(row.get("seed", -9999)) == seed
        ]
        records = [_raw_record(row, origin_s=origin) for row in rows]
        duration = end - origin
        metrics = BenchMetrics.from_records(records, duration)
        slo_result = evaluate_slo(records, slo, duration)
        raw_per_seed.append({
            "seed": seed,
            "metrics": metrics.to_dict(),
            "slo_result": slo_result.to_dict(),
            "primary_value": primary_fn(metrics, slo_result),
        })
    primary_values = [float(row["primary_value"]) for row in raw_per_seed]
    aggregate_metrics = aggregate(primary_values).to_dict()
    aggregate_metrics["primary_metric"] = config.statistical.primary_metric
    member_evals = [member.eval_result for member in members]
    bad_outcome = next(
        (
            result.get("outcome_class")
            for result in member_evals
            if result.get("outcome_class")
            not in {"eval_result", "high_variance_inconclusive"}
        ),
        None,
    )
    outcome = bad_outcome or (
        "high_variance_inconclusive"
        if aggregate_metrics["cv"] > config.statistical.cv_threshold
        else "eval_result"
    )
    result = copy.deepcopy(member_evals[0])
    result.update({
        "source": "real_vllm",
        "outcome_class": outcome,
        "primary_metric": config.statistical.primary_metric,
        "seeds": seeds,
        "sample_count": len(seeds),
        "raw_per_seed_metrics": raw_per_seed,
        "aggregate_metrics": aggregate_metrics,
        "wall_time_s": time.time() - started_at,
        "config_sha256": _canonical_sha(config.to_dict()),
        "command_line": "dual_replica_group",
        "bench_config": config.to_dict(),
        "runner_kind": config.runner.runner_kind,
        "model_served": config.model_to_serve(),
        "rendered_serve_args": config.to_serve_args(),
        "parallel_mode": DUAL_REPLICA,
        "workload_provenance": {
            "source_kind": source_payload.get("source_kind"),
            "scenario": source_payload.get("scenario"),
            "split": source_payload.get("split"),
            "payload_sha256": source_payload.get("payload_sha256"),
            "replay": source_payload.get("replay"),
            "replica_partition_payload_sha256": [
                payload["payload_sha256"] for payload in member_payloads
            ],
        },
        "replica_group": {
            "count": _REPLICA_COUNT,
            "member_run_dirs": [member.local_run_dir for member in members],
            "member_eval_paths": [member.local_eval_path for member in members],
            "member_remote_run_dirs": [member.remote_run_dir for member in members],
        },
    })
    provenance, flags = _combined_plugin_provenance(member_evals)
    result.update(flags)
    if provenance is None:
        result.pop("plugin_provenance", None)
    else:
        result["plugin_provenance"] = provenance
    validity = _group_validity(
        config,
        members,
        member_configs,
        artifacts,
        seeds,
    )
    result["workload_validity"] = validity
    validate_eval_result(result)
    (group_dir / "validity.json").write_text(
        json.dumps(validity, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_remote_replica_group(
    config: BenchConfig,
    *,
    git_sha: str | None,
    single_runner,
    cleanup,
):
    """Execute one logical BenchConfig as two synchronized TP=1 servers."""
    from vllm_evolve.bench.dispatch import RemoteBench

    spec = dict(config.workload.workload_spec or {})
    if spec.get("parallel_mode") != DUAL_REPLICA:
        raise ValueError("replica group requires workload_spec.parallel_mode=dual_replica")
    gpus = [
        part.strip()
        for part in str(config.environment.gpus).split(",")
        if part.strip()
    ]
    if len(gpus) != _REPLICA_COUNT or len(set(gpus)) != _REPLICA_COUNT:
        raise ValueError("dual replica mode requires exactly two distinct GPUs")
    if config.engine.tensor_parallel_size != 1:
        raise ValueError("dual replica mode requires tensor_parallel_size=1")
    if not config.workload.trace_path:
        raise ValueError("dual replica mode requires a materialized replay")
    source_payload = json.loads(
        Path(config.workload.trace_path).read_text(encoding="utf-8")
    )
    expected = source_payload.get("payload_sha256")
    actual = _canonical_sha({
        key: value
        for key, value in source_payload.items()
        if key != "payload_sha256"
    })
    if not expected or expected != actual:
        raise ValueError("source workload payload SHA mismatch")
    payloads = split_scaled_replay_payload(source_payload)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    group_sig = _canonical_sha({
        "config": config.to_dict(),
        "git_sha": git_sha,
        "source_payload_sha256": expected,
    })[:12]
    group_id = f"{stamp}_{group_sig}_{secrets.token_hex(3)}"
    barrier_id = hashlib.sha256(group_id.encode()).hexdigest()[:24]
    group_dir = (
        Path(config.runner.local_artifact_root).expanduser().resolve()
        / "replica_groups"
        / group_id
    )
    inputs_dir = group_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=False)
    member_configs = []
    for index, (gpu, payload) in enumerate(zip(gpus, payloads)):
        workload_path = inputs_dir / f"replica_{index}.json"
        workload_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        member = BenchConfig.from_dict(config.to_dict())
        member.environment.gpus = gpu
        member.runner.port = int(config.runner.port) + index
        member.runner.local_artifact_root = str(
            group_dir / "members" / f"replica_{index}"
        )
        member.workload.trace_path = str(workload_path)
        member.workload.n_requests = int(payload["replay"]["measured_requests"])
        member.workload.concurrency = member.workload.n_requests
        member.workload.arrival_rate_qps = float(
            payload["replay"]["target_qps"]
        )
        member_spec = dict(member.workload.workload_spec)
        member_validity = dict(member_spec.get("validity") or {})
        member_validity["offered_qps"] = member.workload.arrival_rate_qps
        member_spec.update({
            "parallel_mode": REPLICA_MEMBER,
            "replica_member": {
                "group_id": group_id,
                "barrier_id": barrier_id,
                "index": index,
                "count": _REPLICA_COUNT,
                "barrier_timeout_s": min(
                    600.0,
                    max(120.0, float(config.statistical.timeout_s) / 2.0),
                ),
            },
            "validity": member_validity,
        })
        member.workload.workload_spec = member_spec
        member_configs.append(member)
    manifest = {
        "schema_version": 1,
        "source": "real_vllm",
        "parallel_mode": DUAL_REPLICA,
        "group_id": group_id,
        "barrier_id": barrier_id,
        "git_sha": git_sha,
        "aggregate_bench_config": config.to_dict(),
        "member_bench_configs": [member.to_dict() for member in member_configs],
        "source_payload_sha256": expected,
        "member_payload_sha256": [
            payload["payload_sha256"] for payload in payloads
        ],
        "status": "running",
        "members": [],
    }
    manifest_path = group_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    started_at = time.time()
    members = [None] * _REPLICA_COUNT
    failure = None
    with ThreadPoolExecutor(max_workers=_REPLICA_COUNT) as pool:
        future_to_index = {
            pool.submit(single_runner, member, git_sha=git_sha): index
            for index, member in enumerate(member_configs)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                members[index] = future.result()
                manifest["members"].append({
                    "index": index,
                    "status": "completed",
                    "local_run_dir": members[index].local_run_dir,
                    "remote_run_dir": members[index].remote_run_dir,
                })
            except Exception as exc:  # both ports are cleaned below
                failure = exc
                manifest["members"].append({
                    "index": index,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"[:4000],
                })
                for member_config in member_configs:
                    cleanup(
                        member_config.runner.remote,
                        member_config.runner.port,
                    )
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
    if failure is not None or any(member is None for member in members):
        manifest["status"] = "failed"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"dual replica group failed; evidence: {manifest_path}"
        ) from failure

    eval_result = _combine_eval_result(
        config,
        members,
        member_configs,
        payloads,
        source_payload,
        group_dir,
        started_at,
    )
    local_eval = group_dir / "eval_result.json"
    local_eval.write_text(
        json.dumps(eval_result, indent=2) + "\n",
        encoding="utf-8",
    )
    combined_log = group_dir / "serve.log"
    combined_log.write_text(
        "\n".join(
            f"===== replica {index} =====\n{member.log}"
            for index, member in enumerate(members)
        ),
        encoding="utf-8",
    )
    status = {
        "ok": True,
        "state": "completed",
        "parallel_mode": DUAL_REPLICA,
        "duration_s": time.time() - started_at,
        "member_status": [member.status for member in members],
        "gpu_summary": {
            "contaminated": any(
                bool((member.status or {}).get("gpu_summary", {}).get("contaminated"))
                for member in members
            ),
            "per_gpu": {
                gpu: (
                    (member.status or {}).get("gpu_summary", {})
                    .get("per_gpu", {})
                    .get(gpu, {})
                )
                for gpu, member in zip(gpus, members)
            },
        },
    }
    eval_result["remote_cmd"] = "dual_replica_group"
    eval_result["remote_run_dir"] = [
        member.remote_run_dir for member in members
    ]
    eval_result["remote_resource_evidence"] = status["gpu_summary"]
    eval_result["remote_worker_duration_s"] = status["duration_s"]
    local_eval.write_text(
        json.dumps(eval_result, indent=2) + "\n",
        encoding="utf-8",
    )
    (group_dir / "status.json").write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["status"] = "completed"
    manifest["combined_eval_path"] = str(local_eval)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return RemoteBench(
        eval_result=eval_result,
        log=combined_log.read_text(encoding="utf-8"),
        local_eval_path=str(local_eval),
        local_log_path=str(combined_log),
        remote_cmd="dual_replica_group",
        remote_run_dir=",".join(member.remote_run_dir for member in members),
        local_run_dir=str(group_dir),
        status=status,
    )


__all__ = [
    "DUAL_REPLICA",
    "REPLICA_MEMBER",
    "run_remote_replica_group",
    "split_scaled_replay_payload",
]
