# Inferact trace replay (functional E2E)

End-to-end workflow: start vLLM with the GLM Ascend profile, run
`vllm bench serve --agentinfer replay` against the full Inferact Codex trace file, then compare two
completed runs for repeatability.

Case JSON lives under `cases/`. Replay bench defaults come from
[`agentinfer/agentbench/configs/replay_inferact.yaml`](../../../../../../agentinfer/agentbench/configs/replay_inferact.yaml);
E2E only overrides model, base URL (host/port), trace path, task count, and concurrency via CLI.
Other trace-replay suites belong in sibling directories under [`../`](../).

## Module layout

| Module | Role |
| ------ | ---- |
| `run_replay.py` | Pytest: two cold replays + stability assert (`test_inferact_replay`) |
| `conftest.py` | `--test-config-file`, load overrides, `--stability-tolerance` |
| `helpers/replay_config.py` | `ReplayE2EConfig`, `trace-path` resolution, replay CLI, `E2EConfig` for serve |
| `helpers/replay_run.py` | One or two cold `managed_vllm` + replay subprocess cycles |
| `helpers/stability.py` | Two-run comparison (summary + cold-start evidence) |
| `helpers/request_intervals.py` | Adjacent-request timing from `requests.jsonl` |
| [`tests/e2e/helpers/`](../../../helpers/) | **Shared with perf:** `case_loader`, `server`, `benchmark` |

## Trace data

Download [`codex_swebenchpro.json`](https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces/tree/main)
from Hugging Face. The harness passes the **full** file to `--trace-path` (no slicing). Set
`benchmark_params.trace-path` in each case JSON (relative to the repo root, or an absolute path).
Cases default to `"/your/path/codex_swebenchpro.json"`—replace with your downloaded file path before running.

Manual bench (equivalent to what pytest runs):

```bash
vllm bench serve --agentinfer replay \
  --config agentinfer/agentbench/configs/replay_inferact.yaml \
  --trace-path /path/to/codex_swebenchpro.json \
  --base-url http://127.0.0.1:8077 \
  --model glm-5 \
  --task-num 8 --max-concurrency 4 \
  --result-dir test-results/replay/my-run
```

## Run E2E (two cold replays + compare)

```bash
pytest -s -v tests/e2e/function/tracereplay/inferact/run_replay.py::test_inferact_replay \
  --test-config-file tests/e2e/function/tracereplay/inferact/cases/glm52_8x4_inferact_replay.json \
  --stability-tolerance 0.05
```

Case matrix (edit `server_params.model` for your host):

| Case JSON | task-num | max-concurrency |
| --------- | -------- | --------------- |
| `glm52_8x4_inferact_replay.json` | 8 | 4 |
| `glm52_16x8_inferact_replay.json` | 16 | 8 |
| `glm52_24x12_inferact_replay.json` | 24 | 12 |
| `glm52_32x16_inferact_replay.json` | 32 | 16 |

Results land under `test-results/replay/run-<hardware>-local-replay-<tasks>-<concurrency>-<tag>/`
(`summary.json`, `requests.jsonl`, replay artifacts).

Each pytest run performs **two** full cycles (stop/start vLLM between them). Cold start is checked via
`evidence/vllm_metrics_start.prom` (prefix-cache queries at run start must be near zero).

## Repeatability criteria (default 5%)

| Tier | Checks |
| ---- | ------ |
| Exact | `requests.requests`, `tasks.failed`, `requests.input_tokens`, `requests.output_tokens` |
| Primary | Task duration mean / p50 / p95 / p99 within tolerance |
| Secondary | Request latency & TTFT mean / p50 / p95 / p99, prefix-cache hit rate, `run_wall_time_seconds` |
| Intervals | Global adjacent-request interval min / mean / p50 / p90 / p95 / p99 / max from `requests.jsonl` |

Interval definition: `current.started_at - previous.finished_at` per `(session_id, actor_id)` group
(negative values indicate overlap).
