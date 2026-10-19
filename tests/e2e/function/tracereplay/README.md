# Trace replay functional E2E (`tests/e2e/function/tracereplay`)

Cold vLLM + `vllm bench serve --agentinfer replay` repeatability for **Inferact**, **TraceLab**, and **AgentX**.
Requires a GPU/NPU host with vLLM, weights, and trace files. Default CI skips `tests/e2e`.

## Layout

```text
tracereplay/
├── cases/
│   ├── inferact/    # *.json
│   ├── tracelab/
│   └── agentx/
├── test_replay.py   # test_trace_replay[inferact|tracelab|agentx]
├── conftest.py
└── helpers/
```

Shared vLLM/case loading: [`../../helpers/`](../../helpers/).

| File | Role |
| ---- | ---- |
| `test_replay.py` | `ReplaySuiteSpec` per trace family + parametrized pytest |
| `helpers/replay_e2e_config.py` | `ReplayE2EConfig.from_case_file(case, suite)` |
| `helpers/replay_executor.py` | `run_replay_cold_starts` |
| `helpers/replay_compare.py` | Repeatability compare between two run dirs |
| `helpers/replay_repeatability.py` | Shared pytest body |
| `conftest.py` | `--test-config-file`, `--bench-config`, `--cold-runs`, etc. |

`test_replay.py` parametrizes three suites; only the param whose `workload` matches `--test-config-file`
runs—the others **skip**. Prefer `test_replay.py::test_trace_replay[tracelab]`.

## Quick start (pytest)

```bash
pytest -s -v tests/e2e/function/tracereplay/test_replay.py::test_trace_replay[inferact] \
  --test-config-file tests/e2e/function/tracereplay/cases/inferact/glm52_8x4_inferact_replay.json \
  --cold-runs 2 \
  --stability-tolerance 0.05
```

Replace `[inferact]` and `cases/inferact/` with `tracelab` or `agentx` as needed.

| CLI option | Meaning |
| ---------- | ------- |
| `--test-config-file` | **Required.** Case JSON path |
| `--cold-runs` | Cold vLLM + replay cycles (default **2**); compare uses runs **1** and **2** |
| `--stability-tolerance` | Max relative metric gap (default **0.05**) |
| `--task-num` / `--max-concurrency` | Override case JSON load |
| `--bench-config` | Override `benchmark_params.config` (replay bench `--config` YAML) |

Results: `test-results/replay/run-<hardware>-local-replay-<tasks>-<concurrency>-<tag>/` and
`test-results/replay/vllm-logs/`. Each cold cycle stops vLLM between runs.

## Repeatability criteria (default 5%)

Compare **first** and **second** cold run directories.

| Tier | Checks |
| ---- | ------ |
| Informational (non-gating) | Prefix-cache cold evidence; `process_start_time_seconds` (TraceLab may WARN on cold while still PASS) |
| Exact | `requests.requests`, `tasks.failed`, input/output tokens |
| Primary | Task duration mean / p50 / p95 within tolerance; **p99 WARN-only** |
| Secondary | Latency & TTFT mean / p50 / p95 within tolerance; **p99 WARN-only**; prefix-cache hit rate, `run_wall_time_seconds` |
| Intervals | Adjacent-request interval stats from `requests.jsonl` |

Interval: `current.started_at - previous.finished_at` per `(session_id, actor_id)`.

## Inferact

Download [`codex_swebenchpro.json`](https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces/tree/main).
Set `benchmark_params.trace-path` in each case JSON. Bench defaults:
[`replay_inferact.yaml`](../../../agentinfer/agentbench/configs/replay_inferact.yaml).

| Case JSON (`cases/inferact/`) | task-num | max-concurrency |
| ----------------------------- | -------- | --------------- |
| `glm52_8x4_inferact_replay.json` | 8 | 4 |
| `glm52_16x8_inferact_replay.json` | 16 | 8 |
| `glm52_24x12_inferact_replay.json` | 24 | 12 |
| `glm52_32x16_inferact_replay.json` | 32 | 16 |

## TraceLab

Bench defaults:
[`replay_tracelab.yaml`](../../../agentinfer/agentbench/configs/replay_tracelab.yaml). See also
[`docs/en/how-to/run-trace-replay.md`](../../../docs/en/how-to/run-trace-replay.md).

```bash
pytest -s -v tests/e2e/function/tracereplay/test_replay.py::test_trace_replay[tracelab] \
  --test-config-file tests/e2e/function/tracereplay/cases/tracelab/glm52_8x4_tracelab_replay.json \
  --cold-runs 2 --stability-tolerance 0.05
```

| Case JSON (`cases/tracelab/`) | task-num | max-concurrency |
| ----------------------------- | -------- | --------------- |
| `glm52_8x4_tracelab_replay.json` | 8 | 4 |
| `glm52_16x8_tracelab_replay.json` | 16 | 8 |
| `glm52_24x12_tracelab_replay.json` | 24 | 12 |
| `glm52_32x16_tracelab_replay.json` | 32 | 16 |

Manual bench: `vllm bench serve --agentinfer replay --config /path/to/replay_tracelab.yaml --result-dir …`

## AgentX

Bench defaults:
[`replay_agentX.yaml`](../../../agentinfer/agentbench/configs/replay_agentX.yaml). **`trace-path` is required.**

| Case JSON (`cases/agentx/`) | task-num | max-concurrency |
| --------------------------- | -------- | --------------- |
| `glm52_8x4_agentx_replay.json` | 8 | 4 |
| `glm52_16x8_agentx_replay.json` | 16 | 8 |
| `glm52_24x12_agentx_replay.json` | 24 | 12 |
| `glm52_32x16_agentx_replay.json` | 32 | 16 |

## Case JSON ↔ bench CLI

E2E passes `--model`, `--base-url`, `--task-num`, `--max-concurrency`, `--result-dir`. When set in the case,
these `benchmark_params` keys forward to Replay CLI:

| `benchmark_params` | Bench CLI |
| ------------------ | --------- |
| `metrics-url` | `--metrics-url` |
| `endpoint` | `--endpoint` |
| `trace-type` | `--trace-type` |
| `interval-mode` | `--interval-mode` |
| `trace-path` | `--trace-path` |

Fields without Replay CLI (e.g. `chat_template_kwargs`, `tokenizer_base_url`) stay in the YAML from
`benchmark_params.config` or `--bench-config`.

### Workload keys

| Suite | `benchmark_params.workload` |
| ----- | --------------------------- |
| Inferact | `inferact-replay` |
| TraceLab | `tracelab-replay` |
| AgentX | `agentx-replay` |

Common sections: `serve_env` / `server_params` (vLLM serve), `benchmark_params`, `result_root`
(usually `test-results/replay`).
