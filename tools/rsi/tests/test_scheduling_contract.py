from __future__ import annotations

import hashlib
import json

from vllm_evolve.engine.scheduling_contract import load_scheduling_author_contract


def test_frontier_author_contract_is_complete_truthful_and_stable():
    first = load_scheduling_author_contract()
    second = load_scheduling_author_contract()
    assert first == second
    assert "def schedule_batch" in first["skeleton_source"]
    assert "defer_ids" in first["runtime_api_source"]
    assert "mechanism_applicable" in first["capabilities"]["decision_fields"]
    assert "placeholder" in first["capabilities"]["frontier_signal_caveats"]["available_kv_blocks"]
    unhashed = {
        key: value
        for key, value in first.items()
        if key not in {"contract_sha256", "rendered"}
    }
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert first["contract_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
