"""BurstGPT trace adapter (HPMLL/BurstGPT).

Format: UTF-8 CSV with a header row. Two on-disk schemas exist and are detected
by header name, never by position:

* 6-column (BurstGPT_1/_2): ``Timestamp, Model, Request tokens, Response tokens,
  Total tokens, Log Type``
* 8-column (v2.0 / BurstGPT_3): inserts ``Session ID, Elapsed time`` between
  ``Timestamp`` and ``Model``.

``Timestamp`` is a **relative offset in seconds** (integer), not an epoch.
``Request tokens`` = prompt length, ``Response tokens`` = output length. Raw
files include failure rows where ``Response tokens == 0``; ``drop_failures``
removes them (the ``*_without_fails_*`` variants already exclude them).
There is no prefix/KV-reuse field.
"""
from __future__ import annotations

import csv
import hashlib
from collections import deque
from pathlib import Path

from vllm_evolve.bench.datasets.base import TraceDataset, TraceRequest, resolve_column
from vllm_evolve.bench.workload_tokens import prompt_token_ids

_TS = {"timestamp", "arrival_time", "time"}
_PROMPT = {"request_tokens", "prompt_tokens", "input_tokens"}
_OUTPUT = {"response_tokens", "output_tokens", "completion_tokens", "decode_tokens"}


def source_sha256(path: str | Path) -> str:
    """Return the content hash used to bind derived fragments to the official trace."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_valid(path: Path, *, drop_failures: bool = True):
    """Yield ``(valid_index, TraceRequest)`` without loading the full 1.4M-row trace."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"empty BurstGPT file: {path}")
        ts_i = resolve_column(header, _TS)
        p_i = resolve_column(header, _PROMPT)
        o_i = resolve_column(header, _OUTPUT)
        valid_index = 0
        for raw_index, row in enumerate(reader):
            try:
                arrival = float(row[ts_i])
                prompt = int(row[p_i])
                output = int(row[o_i])
            except (IndexError, ValueError):
                continue
            if drop_failures and output <= 0:
                continue
            yield valid_index, TraceRequest(
                arrival_s=arrival,
                prompt_tokens=prompt,
                output_tokens=output,
                request_id=f"burstgpt_{raw_index:07d}",
            )
            valid_index += 1


def load(
    path: str | Path,
    *,
    drop_failures: bool = True,
    max_requests: int | None = None,
    name: str = "burstgpt",
) -> TraceDataset:
    path = Path(path)
    requests: list[TraceRequest] = []
    for _, request in _iter_valid(path, drop_failures=drop_failures):
        if max_requests is not None and len(requests) >= max_requests:
            break
        requests.append(request)
    return TraceDataset(name=name, requests=requests).sorted_by_arrival()


def extract_bursty_fragments(
    path: str | Path,
    *,
    fragment_size: int = 64,
    split_names: tuple[str, ...] = ("train", "validation", "test"),
    band_indices: tuple[int, ...] | None = None,
    total_bands: int | None = None,
    selection_strategy: str = "densest",
    target_mean_total_tokens: float | None = None,
    max_request_total_tokens: int | None = None,
    max_request_output_tokens: int | None = None,
) -> dict[str, TraceDataset]:
    """Pick one dense, contiguous fragment from each chronological source band.

    This is deliberately deterministic and time-ordered: the official trace is divided into
    non-overlapping chronological bands, then one contiguous window is selected inside each band.
    ``densest`` retains the historical shortest-duration rule. ``token_pressure`` selects the
    window whose unmodified mean total-token count is closest to the requested target, using
    duration only as a tie-breaker. The latter is useful for real scheduler experiments where a
    dense but tiny-request fragment can saturate arrival rate without exerting live KV pressure.
    Both modes preserve inter-arrival and token patterns, never shuffle rows, and keep the held-out
    band disjoint from train/validation. Only ``fragment_size`` rows per band are retained while
    scanning the 1.4M-row release.
    """
    source = Path(path)
    if fragment_size < 2:
        raise ValueError("fragment_size must be >= 2")
    if not split_names:
        raise ValueError("at least one split name is required")
    if band_indices is None:
        band_indices = tuple(range(len(split_names)))
    if len(band_indices) != len(split_names):
        raise ValueError("band_indices must have one entry per split name")
    total_bands = int(total_bands or len(split_names))
    if total_bands < 1 or any(index < 0 or index >= total_bands for index in band_indices):
        raise ValueError("band_indices must be inside total_bands")
    if len(set(band_indices)) != len(band_indices):
        raise ValueError("band_indices must be distinct")
    if selection_strategy not in {"densest", "token_pressure"}:
        raise ValueError(
            "selection_strategy must be 'densest' or 'token_pressure'"
        )
    if selection_strategy == "token_pressure":
        if target_mean_total_tokens is None or target_mean_total_tokens <= 0:
            raise ValueError(
                "token_pressure requires a positive target_mean_total_tokens"
            )
        if (
            max_request_total_tokens is None
            or max_request_total_tokens <= 0
        ):
            raise ValueError(
                "token_pressure requires a positive max_request_total_tokens"
            )
        if (
            max_request_output_tokens is not None
            and max_request_output_tokens <= 0
        ):
            raise ValueError("max_request_output_tokens must be positive")

    total = sum(1 for _ in _iter_valid(source))
    if total < fragment_size * total_bands:
        raise ValueError(
            f"BurstGPT trace has {total} valid rows; need at least "
            f"{fragment_size * total_bands} for disjoint fragments"
        )

    bands = []
    for name, index in zip(split_names, band_indices):
        start = (total * index) // total_bands
        end = (total * (index + 1)) // total_bands
        bands.append((name, start, end))

    windows = {name: deque() for name, _, _ in bands}
    token_sums = {name: 0 for name, _, _ in bands}
    max_token_queues: dict[str, deque[tuple[int, int]]] = {
        name: deque() for name, _, _ in bands
    }
    max_output_queues: dict[str, deque[tuple[int, int]]] = {
        name: deque() for name, _, _ in bands
    }
    best: dict[str, tuple[tuple[float, ...], int, list[TraceRequest]]] = {}
    for valid_index, request in _iter_valid(source):
        for name, start, end in bands:
            if not start <= valid_index < end:
                continue
            window = windows[name]
            window.append(request)
            request_tokens = request.prompt_tokens + request.output_tokens
            token_sums[name] += request_tokens
            max_queue = max_token_queues[name]
            while max_queue and max_queue[-1][1] <= request_tokens:
                max_queue.pop()
            max_queue.append((valid_index, request_tokens))
            output_queue = max_output_queues[name]
            while (
                output_queue
                and output_queue[-1][1] <= request.output_tokens
            ):
                output_queue.pop()
            output_queue.append((valid_index, request.output_tokens))
            if len(window) > fragment_size:
                removed_index = valid_index - fragment_size
                removed = window.popleft()
                token_sums[name] -= (
                    removed.prompt_tokens + removed.output_tokens
                )
                if max_queue and max_queue[0][0] == removed_index:
                    max_queue.popleft()
                if output_queue and output_queue[0][0] == removed_index:
                    output_queue.popleft()
            if len(window) != fragment_size:
                break
            duration = window[-1].arrival_s - window[0].arrival_s
            # BurstGPT timestamps are quantized to seconds and can contain a very large
            # simultaneous group.  A zero-duration window erases the temporal burst shape after
            # conversion and makes clairvoyant SJF unbeatable by any causal scheduler.  Keep the
            # densest *positive-duration* contiguous window instead.
            if duration <= 0:
                break
            source_start = valid_index - fragment_size + 1
            if selection_strategy == "densest":
                key = (duration, float(source_start))
            else:
                assert target_mean_total_tokens is not None
                assert max_request_total_tokens is not None
                if (
                    not max_queue
                    or max_queue[0][1] > max_request_total_tokens
                    or (
                        max_request_output_tokens is not None
                        and (
                            not output_queue
                            or output_queue[0][1]
                            > max_request_output_tokens
                        )
                    )
                ):
                    break
                mean_total_tokens = token_sums[name] / fragment_size
                key = (
                    abs(mean_total_tokens - target_mean_total_tokens),
                    duration,
                    float(source_start),
                )
            if name not in best or key < best[name][0]:
                best[name] = (key, source_start, list(window))
            break

    fragments: dict[str, TraceDataset] = {}
    for name, _, _ in bands:
        if name not in best:
            raise RuntimeError(f"could not extract BurstGPT fragment for split {name}")
        _, source_start, rows = best[name]
        fragments[name] = TraceDataset(
            name=(
                f"burstgpt:{name}:{selection_strategy}:"
                f"valid-row-{source_start}"
            ),
            requests=rows,
        ).sorted_by_arrival()
    return fragments


def to_frontier_rows(
    dataset: TraceDataset,
    *,
    target_qps: float,
    session_id: int = 0,
) -> tuple[list[dict], dict]:
    """Convert one real fragment to Frontier's trace-replay contract.

    BurstGPT v1 has no reliable tenant/prefix-reuse signal. Every derived row therefore uses the
    same neutral ``session_id`` and an empty ``block_hash_ids`` value; this function never infers
    cache reuse. Arrival gaps are scaled uniformly to ``target_qps`` (the dataset explicitly permits
    RPS scaling), preserving the fragment's burst shape and request order.
    """
    requests = list(dataset.requests)
    if len(requests) < 2:
        raise ValueError("a Frontier fragment needs at least two requests")
    if target_qps <= 0:
        raise ValueError("target_qps must be positive")
    original_duration = requests[-1].arrival_s - requests[0].arrival_s
    simultaneous_burst = original_duration == 0
    if original_duration < 0:
        raise ValueError("fragment arrivals must be monotonic")
    scaled_duration = (len(requests) - 1) / float(target_qps)
    time_scale = (scaled_duration / original_duration) if original_duration else 0.0
    origin = requests[0].arrival_s
    rows = [{
        "arrived_at": round((r.arrival_s - origin) * time_scale, 9),
        "num_prefill_tokens": int(r.prompt_tokens),
        "num_decode_tokens": int(r.output_tokens),
        "session_id": int(session_id),
        "block_hash_ids": "",
    } for r in requests]
    metadata = {
        "dataset": dataset.name,
        "source_first_request_id": requests[0].request_id,
        "source_last_request_id": requests[-1].request_id,
        "source_start_s": requests[0].arrival_s,
        "source_end_s": requests[-1].arrival_s,
        "source_duration_s": original_duration,
        "source_qps": (len(requests) / original_duration) if original_duration else None,
        "target_qps": float(target_qps),
        "time_scale": time_scale,
        "simultaneous_source_burst": simultaneous_burst,
        "requests": len(requests),
        "tenant_policy": "single neutral session_id=0; BurstGPT prefix reuse not inferred",
    }
    return rows, metadata


def paired_mass_balance_owners(requests: list[dict]) -> list[int]:
    """Assign adjacent arrivals to two replicas with bounded prefix mass skew.

    Each adjacent pair contributes exactly one request to each replica.  The
    orientation is chosen greedily from cumulative prompt, output, and total
    token imbalance, so neither FCFS stream receives a long run of heavier
    requests merely because those shapes occupy one index parity.  Replaying
    the complementary orientation in the next cycle gives each replica every
    official request shape once per cycle pair.
    """
    if len(requests) % 2:
        raise ValueError("paired replica balancing requires an even request count")
    prompt_mass = [0, 0]
    output_mass = [0, 0]
    owners = []
    for pair_start in range(0, len(requests), 2):
        first_request = requests[pair_start]
        second_request = requests[pair_start + 1]

        def assignment_cost(first_owner: int) -> int:
            prompt = list(prompt_mass)
            output = list(output_mass)
            second_owner = 1 - first_owner
            prompt[first_owner] += int(
                first_request.get("num_prompt_tokens") or 0
            )
            output[first_owner] += int(
                first_request.get("num_output_tokens") or 0
            )
            prompt[second_owner] += int(
                second_request.get("num_prompt_tokens") or 0
            )
            output[second_owner] += int(
                second_request.get("num_output_tokens") or 0
            )
            return (
                abs(prompt[0] - prompt[1])
                + abs(output[0] - output[1])
                + abs(
                    (prompt[0] + output[0])
                    - (prompt[1] + output[1])
                )
            )

        owner_zero_cost = assignment_cost(0)
        owner_one_cost = assignment_cost(1)
        pair_index = pair_start // 2
        first_owner = (
            0
            if owner_zero_cost < owner_one_cost
            or (owner_zero_cost == owner_one_cost and pair_index % 2 == 0)
            else 1
        )
        second_owner = 1 - first_owner
        owners.extend((first_owner, second_owner))
        prompt_mass[first_owner] += int(
            first_request.get("num_prompt_tokens") or 0
        )
        output_mass[first_owner] += int(
            first_request.get("num_output_tokens") or 0
        )
        prompt_mass[second_owner] += int(
            second_request.get("num_prompt_tokens") or 0
        )
        output_mass[second_owner] += int(
            second_request.get("num_output_tokens") or 0
        )
    return owners


def expand_scaled_payload(payload: dict) -> list[dict]:
    """Expand compact official-BurstGPT replay cycles for real vLLM.

    Every cycle receives request-unique prompt IDs. Repeating a trace therefore
    cannot invent prefix-cache reuse that BurstGPT v1 does not label.
    """
    requests = list(payload.get("requests") or [])
    replay = dict(payload.get("replay") or {})
    cycles = int(replay.get("cycles") or 1)
    cycle_period = float(replay.get("cycle_period_s") or 0.0)
    partition = dict(payload.get("replica_partition") or {})
    replica_suffix = (
        f"-p{int(partition['index'])}"
        if partition.get("index") is not None else ""
    )
    source_cycles = partition.get("source_cycle_indices")
    if source_cycles is None:
        source_cycles = list(range(cycles))
    else:
        source_cycles = [int(cycle) for cycle in source_cycles]
        if len(source_cycles) != cycles or source_cycles != sorted(source_cycles):
            raise ValueError(
                "replica source cycles must be sorted and match replay.cycles"
            )
    paired_owners = partition.get("base_owner_by_request_index")
    if paired_owners is not None:
        paired_owners = [int(owner) for owner in paired_owners]
        if (
            len(paired_owners) != len(requests)
            or any(owner not in (0, 1) for owner in paired_owners)
        ):
            raise ValueError("paired replica owner map is invalid")
    expanded = []
    for cycle in source_cycles:
        for base_index, request in enumerate(requests):
            if (
                partition.get("strategy") == "cycle_rotated_index_modulo"
                and (
                    (base_index + cycle) % int(partition["count"])
                    != int(partition["index"])
                )
            ):
                continue
            if (
                partition.get("strategy")
                == "cycle_rotated_paired_mass_balance"
                and (
                    (paired_owners[base_index] + cycle) % int(partition["count"])
                    != int(partition["index"])
                )
            ):
                continue
            global_index = cycle * len(requests) + base_index
            prompt_length = int(request["num_prompt_tokens"])
            expanded.append({
                "request_id": (
                    f"{payload.get('split', 'burstgpt')}{replica_suffix}-"
                    f"c{cycle:04d}-r{base_index:04d}"
                ),
                "arrival_s": (
                    cycle * cycle_period + float(request.get("arrival_s") or 0.0)
                ),
                "prompt_token_ids": prompt_token_ids(
                    {"num_prefill_tokens": prompt_length, "block_hash_ids": ""},
                    global_index,
                ),
                "num_prompt_tokens": prompt_length,
                "num_output_tokens": int(request["num_output_tokens"]),
                "source_request_id": request["request_id"],
                "source_cycle": cycle,
            })
    return expanded
