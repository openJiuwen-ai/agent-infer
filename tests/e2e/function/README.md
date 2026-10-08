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
        └── inferact/ # Inferact Codex trace replay (only suite today)
```

| Concern | `perf/` | `function/` (Inferact replay) |
| ------- | ------- | ----------------------------- |
| Bench command | `vllm bench serve --agentinfer run` | `vllm bench serve --agentinfer replay` |
| Default YAML | `swebench_vllm.yaml` (in case) | `replay_inferact.yaml` |
| Pass criteria | Completes; perf compare elsewhere | Cold replay × N + compare run 1 vs 2 |
| Case JSON | `scenario`: baseline / agentinfer | `scenario`: baseline + `workload`: inferact-replay` |

Both reuse [`../helpers/`](../helpers/) for loading case JSON and managing `vllm serve`.

## Inferact layout (`tracereplay/inferact/`)

```text
inferact/
├── test_inferact_replay.py   # pytest entry (test_inferact_replay)
├── conftest.py               # CLI options
├── cases/*.json
└── helpers/
    ├── replay_config.py      # ReplayE2EConfig, replay CLI
    ├── replay_executor.py    # cold vLLM + replay cycles (N times)
    ├── replay_compare.py     # compare two run dirs + report
    └── request_intervals.py
```

| File | Role |
| ---- | ---- |
| `test_inferact_replay.py` | Pytest: N cold replays, then repeatability assert (run 1 vs run 2) |
| `replay_executor.py` | `run_replay_cold_starts(config, cold_runs=N)` |
| `replay_compare.py` | `compare_replay_stability`, `format_stability_report` |

Add another trace family under `tracereplay/<name>/` with its own cases and helpers.

## Trace data

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
| `benchmark_params.trace-path` | Full trace file for `--trace-path` |
| `result_root` | Usually `test-results/replay` |
