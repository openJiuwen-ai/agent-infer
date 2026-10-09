# End-to-end tests (`tests/e2e`)

GPU/NPU-oriented E2E tests live here. They are excluded from default CPU CI (`pytest --ignore=tests/e2e`).

| Subtree | Purpose |
| ------- | ------- |
| [`helpers/`](helpers/) | Shared case JSON loading, vLLM serve lifecycle, completed-run validation |
| [`perf/`](perf/README.md) | Baseline vs AgentInfer performance benchmarks (plan-subagent / BenchKit) |
| [`function/`](function/README.md) | Functional E2E (`tracereplay/inferact`, future trace-replay and serve smoke) |
