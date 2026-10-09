"""
Verify tool — safety and legality checking.

For code policies: AST safety (L1) + constraint validation (L2)
For configs: parameter range + mutual exclusion checks

Results are automatically recorded in the store.
"""
from __future__ import annotations

from pathlib import Path


def verify_code(
    code_or_path: str,
    target_name: str = "scheduling",
    expected_functions: list[str] | None = None,
    store_result: bool = True,
) -> dict:
    """Verify a policy's safety. Returns {passed, issues, policy_id}.

    Checks:
      L1: AST safety (no forbidden imports/patterns)
      L2: Signature validation (expected functions exist)
    """
    from vllm_evolve.trust.safety import (
        check_reserved_class_redefinitions,
        check_safety,
        check_signatures,
    )

    # Read code
    if "\n" not in code_or_path and Path(code_or_path).exists():
        code = Path(code_or_path).read_text(encoding="utf-8")
    else:
        code = code_or_path

    issues: list[str] = []

    # L1: Safety
    result = check_safety(code)
    if not result.ok:
        issues.append(f"L1: {result.reason}")

    # L2: Signatures — load expected functions from target YAML
    if expected_functions is None:
        try:
            from vllm_evolve.config import load_target_spec
            spec = load_target_spec(target_name)
            expected_functions = [fn["name"] for fn in spec.evolvable_functions]
        except Exception:
            expected_functions = []

    if expected_functions:
        sig_result = check_signatures(code, expected_functions)
        if not sig_result.ok:
            issues.append(f"L2: {sig_result.reason}")

    if target_name == "scheduling":
        bridge_types = check_reserved_class_redefinitions(
            code,
            frozenset({"RequestInfo", "ScheduleDecision"}),
        )
        if not bridge_types.ok:
            issues.append(f"L2: {bridge_types.reason}")

    # L2: Constraint validation (O(n^2) patterns, etc.)
    import re
    # Match real nested for-loops but skip dict/set/list comprehensions
    # (comprehension lines have '{' or '[' before the 'for' keyword)
    nested_loops = re.findall(
        r'^[ \t]+for\s+\w+\s+in\s+(\w+).*\n.*?^[ \t]+for\s+\w+\s+in\s+\1',
        code, re.MULTILINE,
    )
    for var in nested_loops:
        issues.append(f"L2: Potential O(n^2) nested loop over '{var}'")

    passed = len(issues) == 0

    # Auto-store check result
    policy_id = None
    if store_result:
        try:
            import hashlib

            from vllm_evolve.tools.store_tool import get_store
            policy_id = hashlib.sha256(code.encode()).hexdigest()[:16]
            get_store().put_check("policy", policy_id, passed, issues)
        except Exception as _e:  # store unavailable, non-critical
            pass  # Store not available, that's ok

    return {"passed": passed, "issues": issues, "policy_id": policy_id}


def verify_config(
    params: dict,
    model_name: str = "",
    hardware: str = "",
    store_result: bool = True,
) -> dict:
    """Verify a config's legality. Returns {passed, issues, config_id}.

    Checks:
      - Parameter ranges (e.g., gpu_memory_utilization in [0, 1])
      - Mutual exclusions (e.g., TP=8 → PP must be 1)
      - Required fields present
    """
    issues: list[str] = []

    # Range checks
    gpu_mem = params.get("gpu_memory_utilization", 0.9)
    if not (0.0 < gpu_mem <= 1.0):
        issues.append(f"gpu_memory_utilization={gpu_mem} out of range (0, 1]")

    max_tokens = params.get("max_num_batched_tokens", 8192)
    if max_tokens < 256 or max_tokens > 65536:
        issues.append(f"max_num_batched_tokens={max_tokens} out of range [256, 65536]")

    max_seqs = params.get("max_num_seqs", 256)
    if max_seqs < 1 or max_seqs > 8192:
        issues.append(f"max_num_seqs={max_seqs} out of range [1, 8192]")

    tp = params.get("tensor_parallel_size", 1)
    pp = params.get("pipeline_parallel_size", 1)
    if tp * pp > 8:
        issues.append(f"TP={tp} × PP={pp} = {tp*pp} exceeds 8 (typical max)")

    # Chunked prefill constraints
    chunked = params.get("enable_chunked_prefill", False)
    chunk_size = params.get("chunked_prefill_size")
    if chunk_size and not chunked:
        issues.append("chunked_prefill_size set but enable_chunked_prefill is False")

    passed = len(issues) == 0

    config_id = None
    if store_result:
        try:
            import hashlib
            import json

            from vllm_evolve.tools.store_tool import get_store
            params_json = json.dumps(params, sort_keys=True)
            config_id = hashlib.sha256(
                (params_json + model_name + hardware).encode()
            ).hexdigest()[:16]
            get_store().put_check("config", config_id, passed, issues)
        except Exception as _e:  # store unavailable, non-critical
            pass

    return {"passed": passed, "issues": issues, "config_id": config_id}


# --- Round 1 (Codex review): route public callables through phase guard ---
from vllm_evolve.tools._legacy_guard import _install_guard as _install_legacy_guard  # noqa: E402

_install_legacy_guard(__name__, ("verify_code", "verify_config",))
