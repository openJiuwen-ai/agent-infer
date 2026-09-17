# AgentInfer Tests

Mirrors [vllm-project/vllm](https://github.com/vllm-project/vllm)
test structure under `tests/agentcache/core/`.

## Test files

### `tests/agentcache/entrypoints/test_serve_profile.py`

Unit tests for the `vllm serve MODEL --agentinfer` takeover in
`agentinfer/agentcache/entrypoints/cli/serve_profile.py` (upstream vLLM modules are stubbed).

| Test | What it verifies |
| ---- | ---------------- |
| `test_bare_flag_pins_async_and_async_bridge` | Bare flag injects async scheduling, async bridge, middleware, and controller factory |
| `test_explicit_async_choice_is_preserved_with_matching_bridge` | `--async-scheduling`/`--no-async-scheduling` are preserved and select the matching bridge |
| `test_upstream_default_async_mode_uses_async_bridge_without_forcing` | Unset mode is pinned async so the bridge matches the engine |
| `test_explicit_scheduler_cls_conflicts` | `--scheduler-cls` combined with `--agentinfer` is rejected |
| `test_user_middleware_order_is_preserved_and_duplicates_removed` | User middleware runs first; duplicates are removed |
| `test_user_additional_config_merges_with_user_priority` | `--additional-config` deep-merges with user priority |
| `test_conflicting_controller_factory_is_rejected` | A different `agentcache.controller_factory` is rejected |
| `test_identical_controller_factory_is_accepted` | The same `controller_factory` merges cleanly |
| `test_invalid_additional_config_json_is_rejected` | Invalid or non-object JSON is rejected |
| `test_run_agentinfer_serve_dispatches_enriched_namespace` | The enriched namespace reaches upstream serve with socket defaulting and the stderr transparency line |
| `test_run_agentinfer_serve_preserves_existing_socket` | An existing `AGENTCACHE_VLLM_LIFECYCLE_SOCKET` is preserved |
| `test_run_agentinfer_serve_requires_the_flag` | Takeover without `--agentinfer` is rejected |

### `tests/agentcache/core/test_agent_scheduler.py`

Unit tests for `AgentAwareScheduler` resolution and the `EngineArgs` patch.

| Test | What it verifies |
| ---- | ---------------- |
| `test_agent_scheduler_is_subclass_of_vllm_scheduler` | `AgentAwareScheduler` extends vllm's `Scheduler` |
| `test_agent_scheduler_is_subclass_of_scheduler_interface` | `AgentAwareScheduler` satisfies `SchedulerInterface` |
| `test_get_scheduler_cls_resolves_agent_scheduler` | `SchedulerConfig.get_scheduler_cls()` resolves the class path |
| `test_get_scheduler_cls_default_is_vllm_scheduler` | Default (unpatched) resolves to vllm's `Scheduler` |
| `test_patch_sets_scheduler_cls_when_none` | `EngineArgs.__post_init__` defaults to `AgentAwareScheduler` when `scheduler_cls` is `None` |
| `test_patch_preserves_explicit_scheduler_cls` | Explicit `scheduler_cls` is not overwritten by the patch |

### `tests/agentcache/core/test_agent_scheduler_e2e.py`

In-process end-to-end test using `agentinfer.LLM`
(wraps `vllm.LLM`) with a tiny model.

| Test | What it verifies |
| ---- | ---------------- |
| `test_agent_scheduler_is_configured` | `scheduler_cls` is set to `AgentAwareScheduler` in the running engine config |
| `test_generation_works` | Text generation produces output |

### `tests/agentcache/core/test_agent_scheduler_serve_e2e.py`

Subprocess end-to-end test via `vllm serve` with a real model.

| Test | What it verifies |
| ---- | ---------------- |
| `test_agent_scheduler_configured` | Server logs contain the `AgentAwareScheduler` log |
| `test_completion` | OpenAI-compatible `/v1/completions` returns text |
| `test_chat_completion` | OpenAI-compatible `/v1/chat/completions` returns a message |

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

### Run unit tests only (no GPU needed)

```bash
pytest tests/agentcache/core/test_agent_scheduler.py -v
```

### Run in-process E2E test (requires GPU, tiny model)

```bash
pytest tests/agentcache/core/test_agent_scheduler_e2e.py -v
```

### Run serve E2E test (requires GPU, real model ~1.3 GB)

```bash
pytest tests/agentcache/core/test_agent_scheduler_serve_e2e.py -v
```
