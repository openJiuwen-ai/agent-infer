"""
Configure tool — Day-0 serving configuration search.

Loads search space from targets/serving_config/search_space.yaml,
generates candidate configs, and manages the search process.
"""
from __future__ import annotations

import itertools
import random

# Default search space (used when no YAML provided)
DEFAULT_SEARCH_SPACE = {
    "serving": {
        "max_num_batched_tokens": [2048, 4096, 8192, 16384],
        "max_num_seqs": [128, 256, 512, 1024],
        "enable_chunked_prefill": [True, False],
        "gpu_memory_utilization": [0.85, 0.90, 0.95],
        "enable_prefix_caching": [True, False],
        "max_model_len": [2048, 4096, 8192, 16384, 32768, 65536, 131072],
        "swap_space_gb": [0, 1, 2, 4, 8, 16],
        "num_gpu_blocks_override": [None, 256, 512, 1024, 2048, 4096],
        "enforce_eager": [True, False],
    },
    "parallelism": {
        "tensor_parallel_size": [1, 2, 4, 8],
        "expert_parallel_size": [1, 2, 4, 8],
        "distributed_executor_backend": ["ray", "mp"],
    },
    "quantization": {
        "dtype": ["float16", "bfloat16"],
        "quantization": [None, "fp8"],
        "kv_cache_dtype": ["auto", "fp8"],
    },
    "execution": {
        "cuda_graph_batch_sizes": [None, [1, 2, 4, 8, 16, 32]],
        "attention_backend": ["flash_attn", "flashinfer", "xformers", "paged_attn"],
        "graph_mode": [True, False],
        "compile_level": [None, "O0", "O1", "O2"],
    },
}

# Default constraints
DEFAULT_CONSTRAINTS = [
    # (param1, val1, param2, allowed_vals2)
    # If TP=8, PP must be 1
    ("tensor_parallel_size", 8, "pipeline_parallel_size", [1]),
    # If chunked_prefill is False, chunked_prefill_size is irrelevant
    ("enable_chunked_prefill", False, "chunked_prefill_size", [None]),
    # enforce_eager=True makes cuda_graph_batch_sizes irrelevant
    ("enforce_eager", True, "cuda_graph_batch_sizes", [None]),
    # graph_mode=False makes compile_level irrelevant (Ascend only)
    ("graph_mode", False, "compile_level", [None]),
]


def flatten_space(space: dict) -> dict[str, list]:
    """Flatten nested search space to {param_name: [values]}."""
    flat = {}
    for group in space.values():
        if isinstance(group, dict):
            flat.update(group)
    return flat


def sample_random(
    space: dict,
    n: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Sample n random configs from the search space."""
    flat = flatten_space(space)
    rng = random.Random(seed)
    configs = []
    for _ in range(n):
        config = {}
        for param, values in flat.items():
            config[param] = rng.choice(values)
        configs.append(config)
    return configs


def sample_grid(
    space: dict,
    max_configs: int = 100,
) -> list[dict]:
    """Generate grid of all combinations (capped at max_configs)."""
    flat = flatten_space(space)
    params = list(flat.keys())
    values = list(flat.values())

    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(params, combo)))
        if len(configs) >= max_configs:
            break
    return configs


def apply_constraints(
    configs: list[dict],
    constraints: list[tuple] | None = None,
) -> list[dict]:
    """Filter out configs that violate constraints."""
    if constraints is None:
        constraints = DEFAULT_CONSTRAINTS

    valid = []
    for cfg in configs:
        ok = True
        for param1, val1, param2, allowed_vals2 in constraints:
            if cfg.get(param1) == val1:
                if cfg.get(param2) not in allowed_vals2:
                    ok = False
                    break
        if ok:
            valid.append(cfg)
    return valid


def generate_candidates(
    space: dict | None = None,
    method: str = "random",
    n: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Generate candidate configs.

    Args:
        space: Search space dict (None = defaults)
        method: "random" or "grid"
        n: Number of candidates
        seed: RNG seed

    Returns:
        List of config dicts, filtered by constraints
    """
    if space is None:
        space = DEFAULT_SEARCH_SPACE

    if method == "grid":
        candidates = sample_grid(space, max_configs=n)
    else:
        candidates = sample_random(space, n=n, seed=seed)

    return apply_constraints(candidates)


# simulate_config removed (M3): DES-simulator-based config evaluation is gone.
# Evaluate vLLM configs on real hardware via the bench harness (`ve bench`).


def export_vllm_yaml(config: dict) -> str:
    """Export a config dict as vLLM-compatible YAML string."""
    import yaml

    # Map our param names to vLLM CLI args
    vllm_config = {}
    for k, v in config.items():
        if v is not None:
            vllm_config[k] = v

    return yaml.dump(vllm_config, default_flow_style=False)


# --- Round 1 (Codex review): route public callables through phase guard ---
from vllm_evolve.tools._legacy_guard import _install_guard as _install_legacy_guard  # noqa: E402

_install_legacy_guard(__name__, (
    "flatten_space", "sample_random", "sample_grid", "apply_constraints",
    "generate_candidates", "export_vllm_yaml",
))
