"""One rendered scheduling author contract shared by context and evolution controllers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONTRACT_VERSION = "scheduling-author-v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_scheduling_author_contract(*, backend: str = "frontier") -> dict:
    """Return the exact source surfaces plus truthful backend capability caveats.

    The legacy target skeleton remains the static signature authority.  Frontier's runtime decision
    API is included verbatim so an author can see the extra fields the bridge actually accepts.
    Keeping both in one hashed payload makes drift visible instead of silently hiding it behind the
    static verifier.
    """
    if backend != "frontier":
        raise ValueError(f"unsupported scheduling author backend: {backend}")
    root = _repo_root()
    skeleton_path = root / "targets" / "scheduling" / "skeleton.py"
    runtime_path = root / "integrations" / "frontier" / "ve_policy_api.py"
    skeleton = skeleton_path.read_text(encoding="utf-8")
    runtime_api = runtime_path.read_text(encoding="utf-8")
    payload = {
        "version": CONTRACT_VERSION,
        "backend": backend,
        "schedule_signature_authority": str(skeleton_path.relative_to(root)),
        "runtime_decision_authority": str(runtime_path.relative_to(root)),
        "skeleton_source": skeleton,
        "runtime_api_source": runtime_api,
        "capabilities": {
            "request_fields": [
                "request_id",
                "arrival_time_s",
                "remaining_prompt_tokens",
                "num_output_tokens",
                "prefix_cached_tokens",
                "kv_blocks_used",
                "num_preemptions",
                "is_prefill",
                "has_prefix_hint",
                "session_id",
                "generated_output_tokens",
                "waiting_age_s",
                "remaining_output_tokens",
            ],
            "decision_fields": [
                "prefill_batch",
                "decode_batch",
                "preempt_ids",
                "defer_ids",
                "mechanism_applicable",
            ],
            "frontier_signal_caveats": {
                "available_kv_blocks": "placeholder in the current Frontier bridge",
                "prefix_cache_hit_rate": "0.0 in the current Frontier bridge",
                "kv_blocks_used": "not a live per-request Frontier signal",
            },
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["contract_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["rendered"] = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    return payload


__all__ = ["CONTRACT_VERSION", "load_scheduling_author_contract"]
