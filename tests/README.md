# AgentInfer Tests

Mirrors [vllm-project/vllm](https://github.com/vllm-project/vllm)
test structure under `tests/agentcache/core/`.

## Test files

### `tests/agentcache/entrypoints/test_serve_profile.py`

Unit tests for the `vllm serve MODEL --agentinfer` takeover in
`agentinfer/agentcache/entrypoints/cli/serve_profile.py` (upstream vLLM modules are stubbed).

| Test | What it verifies |
| ---- | ---------------- |
| `test_bare_flag_pins_async_and_async_bridge` | Bare flag injects async scheduling, async bridge, and middleware |
| `test_explicit_async_choice_is_preserved_with_matching_bridge` | `--async-scheduling`/`--no-async-scheduling` are preserved and select the matching bridge |
| `test_upstream_default_async_mode_uses_async_bridge_without_forcing` | Unset mode is pinned async so the agent-aware scheduler matches the engine |
| `test_explicit_scheduler_cls_conflicts` | `--scheduler-cls` combined with `--agentinfer` is rejected |
| `test_user_middleware_order_is_preserved_and_duplicates_removed` | User middleware runs first; duplicates are removed |
| `test_user_additional_config_merges_with_user_priority` | `--additional-config` deep-merges with user priority |
| `test_invalid_additional_config_json_is_rejected` | Invalid or non-object JSON is rejected |
| `test_run_agentinfer_serve_dispatches_enriched_namespace` | The enriched namespace reaches upstream serve with socket defaulting and the stderr transparency line |
| `test_run_agentinfer_serve_preserves_existing_socket` | An existing `AGENTCACHE_VLLM_LIFECYCLE_SOCKET` is preserved |
| `test_run_agentinfer_serve_requires_the_flag` | Takeover without `--agentinfer` is rejected |
| `test_run_agentinfer_serve_applies_upstream_cli_env_setup` | The takeover invokes the upstream `cli_env_setup()` before parsing |
| `test_lifecycle_socket_env_defaults_to_config_path` | An unset socket env defaults to `agentcache.lifecycle_socket_path` from `--additional-config` |
| `test_lifecycle_socket_env_and_config_conflict_is_rejected` | An env socket conflicting with a config socket is rejected |
| `test_transparency_line_omits_user_config_secrets` | The transparency line reports only injected `agentcache` keys, never arbitrary user config |

### `tests/agentcache/entrypoints/test_serve_flag_e2e.py`

GPU e2e tests for `vllm serve MODEL --agentinfer` through the installed
`vllm` entrypoint (requires a vLLM environment; skip-marked as `gpu_test`).

| Test | What it verifies |
| ---- | ---------------- |
| `test_injection_banner_logged` | Server logs carry the `[agentinfer] --agentinfer injected:` banner with the async agent-aware scheduler and lifecycle middleware |
| `test_lifecycle_socket_bound` | The configured `AGENTCACHE_VLLM_LIFECYCLE_SOCKET` is bound while serving |
| `test_completion` | A `/v1/completions` request round-trips through the serving path |
| `test_chat_completion` | A `/v1/chat/completions` request round-trips through the serving path |
| `test_conflicting_scheduler_cls_exits_before_engine_startup` | `--agentinfer --scheduler-cls ...` exits with code 2 before engine startup |

## How to run

### Prerequisites

- Python 3.10+, CUDA GPU with sufficient memory
- `vllm==0.22.1` installed

```bash
pip install -e ".[dev]"
```

### Run all tests

```bash
pytest tests/agentcache/ tests/agentbench/ -v
```

### Run E2E performance benchmarks (GPU / NPU, real model)

See [`tests/e2e/README.md`](e2e/README.md). These cases are not part of CPU CI.
