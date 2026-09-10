# E2E Performance Tests

Baseline and AgentInfer cases are **fully independent** JSON files. Each file contains its own
environment variables, vLLM serve parameters, and benchmark parameters.

Run benchmarks with pytest (one JSON per invocation):

```bash
pytest -s -v tests/e2e/run_benchmark.py \
  --test-config-file tests/e2e/cases/baseline/glm52_8x4_plan_subagent.json
```

## Case file layout

| Path | Meaning |
| ---- | ------- |
| `tests/e2e/cases/baseline/glm52_{8x4,16x8,24x12,32x16}_plan_subagent.json` | Baseline vLLM serve + plan-subagent benchmark |
| `tests/e2e/cases/agentinfer/glm52_{8x4,16x8,24x12,32x16,480x12}_plan_subagent.json` | AgentInfer variant + plan-subagent benchmark (480x12 = long stability soak) |

## Shared schema

```json
{
  "test_name": "glm52_8x4_plan_subagent",
  "scenario": "agentinfer",
  "description": "…",
  "serve_env": { "…": "…" },
  "server_params": {
    "model": "/home/models/GLM-5.2-w4a8c8",
    "middleware": ["…"],
    "serve_args": { "tensor-parallel-size": 16 }
  },
  "result_root": "test-results/agentinfer",
  "benchmark_params": {
    "config": "agentinfer/agentbench/configs/swebench_vllm.yaml",
    "prepare_dataset": "swebench",
    "prepare_output_dir": "agentinfer/agentbench/data/swebench",
    "host": "127.0.0.1",
    "port": 8000,
    "task-num": 8,
    "max-concurrency": 4
  }
}
```

| Field | Purpose |
| ----- | ------- |
| `scenario` | `baseline` or `agentinfer` (legacy JSON key `arm` still accepted) |
| `mark` | Optional metadata (e.g. hardware notes for CI job selection) |
| `wait_for_vllm_ready` | Optional. Poll `/v1/models` after vLLM start (default: `true`; omit in JSON) |
| `serve_env` | vLLM serve subprocess environment variables |
| `server_params` | Model path, middleware, and `vllm serve` CLI flags |
| `benchmark_params` | BenchKit `run` workload and dataset prepare settings |
| `benchmark_params.benchmark-run-as-user` | E2E-only. For `plan-subagent`, wrap bench with `sudo -u USER` when pytest runs as root; ignored for `single` |

## Usage

```bash
export ANTHROPIC_AUTH_TOKEN=agentinfer-local-smoke

# Baseline
pytest -s -v tests/e2e/run_benchmark.py \
  --test-config-file tests/e2e/cases/baseline/glm52_8x4_plan_subagent.json

# AgentInfer
pytest -s -v tests/e2e/run_benchmark.py \
  --test-config-file tests/e2e/cases/agentinfer/glm52_8x4_plan_subagent.json

# Override workload without editing JSON
pytest -s -v tests/e2e/run_benchmark.py \
  --test-config-file tests/e2e/cases/agentinfer/glm52_8x4_plan_subagent.json \
  --task-num 16 --max-concurrency 8
```

Logs stream to the terminal; vLLM output is also tee'd under `{result_root}/vllm-logs/`.
BenchKit stdout (including `Results:` and the tqdm bar) is suppressed during the run; the pytest
line `[benchmark:<scenario>] completed: ... dir=...` reports where artifacts were written.

Compare two finished runs manually with BenchKit:

```bash
vllm bench serve --agentinfer compare \
  --baseline test-results/vllm/run-... \
  --candidate test-results/agentinfer/run-...
```

## Module layout

| Module | Role |
| ------ | ---- |
| `run_benchmark.py` | Pytest entry: `test_benchmark_completes` |
| `helpers/case_loader.py` | JSON parsing, `E2EPerfConfig`, argv/env builders, `sudo -u` wrapping |
| `helpers/server.py` | vLLM lifecycle, ready wait, port cleanup, log streaming |
| `helpers/benchmark.py` | Dataset prepare, BenchKit run orchestration, run validation |
