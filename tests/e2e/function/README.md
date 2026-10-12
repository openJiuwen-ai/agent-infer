# E2E functional tests (`tests/e2e/function`)

Correctness-oriented GPU/NPU E2E checks live here. They are **not** performance A/B suites (`tests/e2e/perf/`).

Default CI skips all of `tests/e2e` (`pytest --ignore=tests/e2e`). Run these on a machine with vLLM,
weights, and (for replay) trace files.

## Position in the repo

```text
tests/e2e/
├── helpers/          # shared: case JSON, vLLM lifecycle, validate_completed_run
├── perf/             # BenchKit plan-subagent; baseline vs AgentInfer
└── function/         # replay repeatability, future serve smoke, etc.
    └── tracereplay/
        ├── helpers/  # shared replay E2E (config, executor, compare, pytest options)
        ├── specs.py  # inferact / tracelab / agentx ReplaySuiteSpec
        ├── inferact/ # cases + thin pytest entry
        ├── tracelab/
        └── agentx/
```

| Concern | `perf/` | `function/` (Inferact / TraceLab / AgentX replay) |
| ------- | ------- | --------------------------------------------------- |
| Bench command | `vllm bench serve --agentinfer run` | `vllm bench serve --agentinfer replay` |
| Default YAML | `swebench_vllm.yaml` (in case) | `replay_inferact.yaml` / `replay_tracelab.yaml` / `replay_agentX.yaml` |
| Pass criteria | Completes; perf compare elsewhere | Cold replay × N + compare run 1 vs 2 |
| Case JSON | `scenario`: baseline / agentinfer | `scenario`: baseline + `workload`: `*-replay` |

Both reuse [`../helpers/`](../helpers/) for loading case JSON and managing `vllm serve`.

## Shared replay E2E (`tracereplay/helpers/` + `specs.py`)

| File | Role |
| ---- | ---- |
| `specs.py` | `INFERACT_REPLAY_SUITE`, `TRACELAB_REPLAY_SUITE`, `AGENTX_REPLAY_SUITE` (workload, default YAML, trace-path policy) |
| `helpers/replay_e2e_config.py` | `ReplayE2EConfig.from_case_file(case, suite)` |
| `helpers/replay_executor.py` | `run_replay_cold_starts` |
| `helpers/replay_compare.py` | Repeatability compare between two run dirs |
| `helpers/replay_repeatability.py` | Shared pytest body |
| `helpers/conftest_options.py` | `--test-config-file`, `--bench-config`, etc. |

`tracereplay/conftest.py` registers shared CLI options once (avoids duplicate `--test-config-file` across suites).
Each suite under `tracereplay/<name>/` keeps `cases/*.json` and a thin `test_*_replay.py`.
Add a new trace family: one entry in `specs.py`, then `<name>/cases/` + pytest shim.

## TraceLab layout (`tracereplay/tracelab/`)

Same pytest/cold-start/repeatability flow as Inferact; bench defaults come from
[`agentinfer/agentbench/configs/replay_tracelab.yaml`](../../agentinfer/agentbench/configs/replay_tracelab.yaml)
(`replay.trace_type: tracelab`, `/v1/chat/completions`, zero calibration tolerance).

### Manual replay — TraceLab

```bash
export REPLAY_RESULT=path/to/tracelab_test

vllm bench serve --agentinfer replay \
  --config /path/to/replay_tracelab.yaml \
  --result-dir "$REPLAY_RESULT"
```

Copy or extend the packaged YAML on your host (model, `base_url`, `trace_path`, `task_num`,
`max_concurrency`). vLLM **serve** uses the same case JSON `serve_env` / `server_params` as Inferact E2E.

### Run TraceLab replay E2E

```bash
pytest -s -v tests/e2e/function/tracereplay/tracelab/test_tracelab_replay.py::test_tracelab_replay \
  --test-config-file tests/e2e/function/tracereplay/tracelab/cases/glm52_8x4_tracelab_replay.json \
  --cold-runs 2 \
  --stability-tolerance 0.05
```

| Case JSON | task-num | max-concurrency |
| --------- | -------- | --------------- |
| `glm52_8x4_tracelab_replay.json` | 8 | 4 |
| `glm52_16x8_tracelab_replay.json` | 16 | 8 |
| `glm52_24x12_tracelab_replay.json` | 24 | 12 |
| `glm52_32x16_tracelab_replay.json` | 32 | 16 |

Case JSON uses `benchmark_params.workload`: `tracelab-replay`. Set `benchmark_params.trace-path` to a
normalized TraceLab JSONL file, or omit / set `null` in a custom YAML so Replay downloads the pinned
`UW-SyFI/TraceLab` snapshot for `task-num` sessions (see
[`docs/en/how-to/run-trace-replay.md`](../../docs/en/how-to/run-trace-replay.md)).

## AgentX layout (`tracereplay/agentx/`)

Same pytest/cold-start/repeatability flow as TraceLab; bench defaults come from
[`agentinfer/agentbench/configs/replay_agentX.yaml`](../../agentinfer/agentbench/configs/replay_agentX.yaml)
(`replay.trace_type: agentX`). Host templates often set `backend.endpoint: /v1/completions` and
AgentX-specific replay fields (e.g. `max_inflight_requests`) in a copied YAML such as
`replay_agentX_test.yaml`.

### Manual replay — AgentX

```bash
export REPLAY_RESULT=/home/y00623537/test-result/1009/agentx_test_8_4_1st

vllm bench serve --agentinfer replay \
  --config /home/y00623537/trace-file/replay_agentX_test.yaml \
  --result-dir "$REPLAY_RESULT"
```

vLLM **serve** matches Inferact/TraceLab E2E (`serve_env` / `server_params` in the case JSON).

### Run AgentX replay E2E

```bash
pytest -s -v tests/e2e/function/tracereplay/agentx/test_agentx_replay.py::test_agentx_replay \
  --test-config-file tests/e2e/function/tracereplay/agentx/cases/glm52_8x4_agentx_replay.json \
  --cold-runs 2 \
  --stability-tolerance 0.05
```

| Case JSON | task-num | max-concurrency |
| --------- | -------- | --------------- |
| `glm52_8x4_agentx_replay.json` | 8 | 4 |
| `glm52_16x8_agentx_replay.json` | 16 | 8 |
| `glm52_24x12_agentx_replay.json` | 24 | 12 |
| `glm52_32x16_agentx_replay.json` | 32 | 16 |

Case JSON uses `benchmark_params.workload`: `agentx-replay`. **`trace-path` is required** (AgentX JSONL
snapshot). Optional `benchmark_params.config` or `--bench-config` points at your host YAML (see Inferact
CLI table for `--bench-config`).

Replay execution for `trace_type: agentX` may still raise `NotImplementedError` until the adapter lands in
`agentinfer`; the E2E harness is ready for repeatability runs once bench completes.

## Case JSON (AgentX)

| Section | Role |
| ------- | ---- |
| `serve_env` / `server_params` | Same vLLM serve shape as other replay E2E cases |
| `benchmark_params.workload` | Must be `agentx-replay` |
| `benchmark_params.config` | Optional replay bench YAML (default: `replay_agentX.yaml`) |
| `benchmark_params.trace-path` | Required AgentX JSONL for `--trace-path` |
| `result_root` | Usually `test-results/replay` |

## Trace data (Inferact)

Download [`codex_swebenchpro.json`](https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces/tree/main)
from Hugging Face. Set `benchmark_params.trace-path` in each case JSON (absolute path or relative to
repo root). Cases default to `"/your/path/codex_swebenchpro.json"`.

Replay bench defaults:
[`agentinfer/agentbench/configs/replay_inferact.yaml`](../../agentinfer/agentbench/configs/replay_inferact.yaml).
E2E overrides model, base URL, trace path, task count, and concurrency via CLI.

## Run Inferact replay E2E

```bash
pytest -s -v tests/e2e/function/tracereplay/inferact/test_inferact_replay.py::test_inferact_replay \
  --test-config-file tests/e2e/function/tracereplay/inferact/cases/glm52_8x4_inferact_replay.json \
  --cold-runs 2 \
  --stability-tolerance 0.05
```

| CLI option | Meaning |
| ---------- | ------- |
| `--test-config-file` | **Required.** Case JSON path |
| `--cold-runs` | Cold vLLM + replay cycles (default **2**); compare uses runs **1** and **2** |
| `--stability-tolerance` | Max relative metric gap (default **0.05**) |
| `--task-num` / `--max-concurrency` | Override case JSON load |
| `--bench-config` | Override `benchmark_params.config` (replay bench `--config` YAML) |

Case matrix (edit `server_params.model` on your host):

| Case JSON | task-num | max-concurrency |
| --------- | -------- | --------------- |
| `glm52_8x4_inferact_replay.json` | 8 | 4 |
| `glm52_16x8_inferact_replay.json` | 16 | 8 |
| `glm52_24x12_inferact_replay.json` | 24 | 12 |
| `glm52_32x16_inferact_replay.json` | 32 | 16 |

Results: `test-results/replay/run-<hardware>-local-replay-<tasks>-<concurrency>-<tag>/` plus
`test-results/replay/vllm-logs/`. Each cold cycle stops vLLM between runs; cold start is checked via
`evidence/vllm_metrics_start.prom`.

## Repeatability criteria (default 5%)

Compare **first** and **second** cold run directories.

| Tier | Checks |
| ---- | ------ |
| Exact | `requests.requests`, `tasks.failed`, input/output tokens |
| Primary | Task duration mean / p50 / p95 / p99 within tolerance |
| Secondary | Latency & TTFT percentiles, prefix-cache hit rate, `run_wall_time_seconds` |
| Intervals | Adjacent-request interval stats from `requests.jsonl` |

Interval: `current.started_at - previous.finished_at` per `(session_id, actor_id)`.

## Case JSON (Inferact)

| Section | Role |
| ------- | ---- |
| `serve_env` / `server_params` | vLLM serve (Inferact E2E: `middleware: []`, `scenario: baseline`) |
| `benchmark_params.workload` | Must be `inferact-replay` |
| `benchmark_params.config` | Optional replay bench YAML for `--config` (default in code: `replay_inferact.yaml`) |
| `benchmark_params.trace-path` | Full trace file for `--trace-path` |
| `result_root` | Usually `test-results/replay` |

On a host-specific bench template (e.g. `/home/.../replay_tracelab_test.yaml`), set `config` in the case JSON
or pass `--bench-config`. E2E still passes `--model`, `--base-url`, `--task-num`, and `--max-concurrency`, which
override the same keys in that YAML when the CLI supplies them.

## Case JSON (TraceLab)

| Section | Role |
| ------- | ---- |
| `serve_env` / `server_params` | Same vLLM serve shape as Inferact E2E |
| `benchmark_params.workload` | Must be `tracelab-replay` |
| `benchmark_params.config` | Optional replay bench YAML (default in code: `replay_tracelab.yaml`) |
| `benchmark_params.trace-path` | TraceLab JSONL for `--trace-path`; omit to rely on YAML `trace_path: null` + download |
| `result_root` | Usually `test-results/replay` |
