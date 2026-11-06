# End-to-end tests (`tests/e2e`)

GPU/NPU-oriented E2E tests live here. They are excluded from default CPU CI (`pytest --ignore=tests/e2e`).

Shared helpers for case JSON and vLLM lifecycle: [`helpers/`](helpers/).

## Suites

| Subtree | Purpose |
| ------- | ------- |
| [`perf/`](perf/README.md) | Baseline vs AgentInfer performance benchmarks (plan-subagent / BenchKit) |
| [`function/`](function/README.md) | Functional E2E (`agentbench/dsh`, `tracereplay/inferact`, TraceLab, AgentX) |

`perf/` and `function/` are independent: different bench commands, pass criteria, and case JSON schemas.
