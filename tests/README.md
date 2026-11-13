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
`vllm` entrypoint (requires a vLLM environment; marked `L20`, `cards_1`, and `ready`).

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

### Run E2E tests (GPU / NPU, real model)

End-to-end tests live under [`tests/e2e/`](e2e/README.md) (performance benchmarks and functional suites).
They have no ready lane and are not selected by default CPU CI (`-m "cpu and ready"`).

| Suite | Entry doc |
| ----- | --------- |
| Overview | [`tests/e2e/README.md`](e2e/README.md) |
| Performance (baseline vs AgentInfer) | [`tests/e2e/perf/README.md`](e2e/perf/README.md) |
| Functional (trace replay, etc.) | [`tests/e2e/function/README.md`](e2e/function/README.md) |

### Run and Replay CPU smoke tests

Run 模式和 Replay 共用独立进程 Mock Server。
[Replay CI 说明](agentbench/replay/README.md) 包含双轮冷启动矩阵、后端契约、
依赖与故障测试、执行命令和失败产物说明。
CPU CI 沿用现有 Python 3.10/3.12 任务，通过 `cpu and ready` 选择单元测试及 Run / Replay 冒烟用例。
单独运行冒烟用例时，使用下方的具体文件路径加 `-m "cpu and ready"` 筛选，不再使用额外的 smoke 标签。

## Pytest labels

See [RFC #81](https://github.com/openJiuwen-ai/agent-infer/issues/81).
Labels are registered only in `pyproject.toml`. Each dimension is optional,
with at most one value per dimension:

| Dimension | Values |
| --- | --- |
| Model | `cpu`, `L20`, `A3` |
| Card count | `cards_1`, `cards_2`, `cards_16` |
| CI lane | `ready`, `nightly` (reserved) |

Card counts describe the number of accelerators, not device indices. CPU tests
have no card-count label. Add other positive card counts to the registry when
needed. Scenes are selected through directories (`function/`, `perf/`, and any
future `acc/`), with no scene markers.

Unit tests without model or card-count labels default to `cpu`, adding `ready`
only when no CI lane is explicit. Explicit hardware prevents these defaults:
`agentcache/core` has only `cpu`; the serve smoke has `L20`, `cards_1`, and
`ready`. Smaller test groups are selected by file path. CPU labels describe
hardware; real DSH tests still require DSH and its native sandbox.

E2E case JSON retains the hardware object:

```json
"mark": [
  {"hardware_marks": {"res": {"npu": "A3"}, "num_cards": 16}}
]
```

This generates two independent markers, `A3` and `cards_16`. CUDA uses
`res: {"cuda": "L20"}`; CPU uses the string `cpu`. Optional lane labels are
strings in the same list. Unknown labels, conflicting dimension values,
malformed hardware objects, and CPU/card-count combinations fail collection.
Hardware model descriptions register their platform with `[cuda]` or `[npu]`;
this is registry metadata, not an additional pytest marker. Platform/model
mismatches are rejected. `timeout` is registered as a mechanism marker so
collection works without pytest-timeout; enforcement still requires the plugin.
The case id includes its relative directory to distinguish baseline and
AgentInfer files with the same name. `--test-config-file` selects one case.

```bash
pytest -q -m "cpu and ready" tests
pytest -sv -m "L20 and cards_1 and ready" tests/agentcache/entrypoints/test_serve_flag_e2e.py
pytest --collect-only -q -m "A3 and cards_16" tests/e2e/function/tracereplay/
pytest --collect-only -q -m "A3 and cards_16" tests/e2e/perf/run_benchmark.py
```

The 12 replay and nine perf cases have no lane and stay out of ready CI.
`cards_2` supports the DSH cases proposed in PR #75; this baseline does not
contain those cases or their CI step. No nightly pipeline is introduced.
`inferrsi/tests/` remains outside this marker policy. A single-case override
must name an existing JSON in a suite's cases tree and match a collected suite;
invalid overrides fail instead of silently skipping an empty parameter set.

### Select smoke and real-runtime tests by path

All three groups use `cpu and ready`; their file paths distinguish them.
These commands preserve the previous groups without custom sublabels:

```bash
# Benchmark smoke
pytest -q -m "cpu and ready" tests/agentbench/test_benchmark_smoke.py

# Replay HTTP smoke
pytest -q -m "cpu and ready" \
  tests/agentbench/replay/test_cold_start.py \
  tests/agentbench/replay/test_http_dependencies.py \
  tests/agentbench/replay/test_validation.py

# Real DSH runtime and sandbox
pytest -q -m "cpu and ready" \
  tests/agentbench/agents/dsh/test_dsh_benchmark_integration.py \
  tests/agentbench/agents/dsh/test_dsh_sandbox_integration.py
```
