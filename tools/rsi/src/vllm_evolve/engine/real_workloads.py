"""Materialize simulator scenarios as exact-token real-vLLM replay inputs.

The source rows come from the same ``build_scenarios`` function used by local
Frontier search.  Real requests carry raw token IDs, so prompt lengths seen by
vLLM exactly match the simulated lengths.  Official BurstGPT rows receive
request-unique prefixes; synthetic stress rows preserve only their explicitly
labeled shared prefix blocks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vllm_evolve.bench.workload_tokens import prompt_token_ids
from vllm_evolve.engine.local_frontier_evolve import Scenario, build_scenarios


def _canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def scenario_payload(scenario: Scenario) -> dict:
    requests = []
    for index, row in enumerate(scenario.rows):
        requests.append({
            "request_id": f"{scenario.name}-r{index:04d}",
            "arrival_s": float(row["arrived_at"]),
            "prompt_token_ids": prompt_token_ids(row, index),
            "num_prompt_tokens": int(row["num_prefill_tokens"]),
            "num_output_tokens": int(row["num_decode_tokens"]),
            "session_id": int(row.get("session_id") or 0),
            "source_prefix_blocks": str(row.get("block_hash_ids") or ""),
        })
    core = {
        "schema_version": 1,
        "scenario": scenario.name,
        "split": scenario.split,
        "source_kind": scenario.source_kind,
        "slo_ttft_ms": scenario.slo_ttft_ms,
        "max_num_seqs": scenario.max_num_seqs,
        "enable_prefix_caching": scenario.enable_prefix_caching,
        "source_rows_sha256": _canonical_sha(scenario.rows),
        "metadata": scenario.metadata,
        "requests": requests,
    }
    return {**core, "payload_sha256": _canonical_sha(core)}


def materialize(
    burstgpt_path: str | Path,
    out_dir: str | Path,
    *,
    fragment_size: int = 32,
    slo_ttft_ms: float = 200.0,
    max_num_seqs: int = 4,
    stress_prefix_blocks: int = 2,
) -> dict:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scenarios, source_manifest = build_scenarios(
        burstgpt_path,
        fragment_size=fragment_size,
        slo_ttft_ms=slo_ttft_ms,
        max_num_seqs=max_num_seqs,
        stress_prefix_blocks=stress_prefix_blocks,
        splits=("test",),
    )
    wanted = {"burstgpt_test", "stress_test_moderate", "stress_test_severe"}
    files = {}
    for scenario in scenarios:
        if scenario.name not in wanted:
            continue
        payload = scenario_payload(scenario)
        path = out / f"{scenario.name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        files[scenario.name] = {
            "path": str(path),
            "payload_sha256": payload["payload_sha256"],
            "requests": len(payload["requests"]),
            "source_rows_sha256": payload["source_rows_sha256"],
        }
    missing = sorted(wanted - files.keys())
    if missing:
        raise RuntimeError(f"missing real-vLLM scenarios: {missing}")
    manifest = {
        "schema_version": 1,
        "source": "local_frontier_evolve.build_scenarios",
        "burstgpt": source_manifest,
        "files": files,
        "honesty": (
            "BurstGPT has no prefix labels and receives request-unique token prefixes. "
            "Only synthetic stress rows preserve explicitly generated shared prefixes."
        ),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm_evolve.engine.real_workloads")
    parser.add_argument("--burstgpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fragment-size", type=int, default=32)
    parser.add_argument("--slo-ttft-ms", type=float, default=200.0)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--stress-prefix-blocks", type=int, default=2)
    args = parser.parse_args(argv)
    manifest = materialize(
        args.burstgpt,
        args.out,
        fragment_size=args.fragment_size,
        slo_ttft_ms=args.slo_ttft_ms,
        max_num_seqs=args.max_num_seqs,
        stress_prefix_blocks=args.stress_prefix_blocks,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
