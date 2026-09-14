# AgentBench

AgentBench separates orchestration (`benchkit/`), Claude Code/JiuwenSwarm/DSH execution (`agents/`), and transparent
request observation (`request_proxy/`). Baseline and Scheduler candidate runs both target vLLM through the same
Request Proxy.

Start with the [Benchmark Quickstart](../../docs/en/how-to/run-benchmark.md). Module boundaries and invariants are
documented under [`docs/design/module/benchmarking`](../../docs/en/explanation/benchmark-methodology.md).

## Trace Replay scope

Trace Replay primarily accepts a captured `requests.jsonl`. Without storing or reading original Prompt plaintext, it
reconstructs replayable Sessions, Actors, request kinds, token targets, context relationships, dependencies, historical
start offsets, and synthetic shared-prefix structure. The current validation workflow executes the same trace twice,
restarting vLLM before each Replay and confirming Prefix Cache counters start at zero, then compares the two finalized
`summary.json` files for cold-start reproducibility.

Replay is a workload and performance reconstruction path, not a second Claude Code execution. The current synthetic
Prompt does not fully reproduce Claude Code semantics, complete Tool schemas, `tool_use`/`tool_result` blocks, natural
stopping behavior, or task correctness. Its metrics may still differ from an actual Claude Code run and must not be
reported as exact execution parity.

## Migration provenance

AgentBench and its focused tests are synchronized from JiusiServe/AgentInfer commit
`abcaaba9b8e4960ba5e29add24c1a7b99d6c14ad`. The initial import used
`af93e654606d5f5365a64aaacfc5cb63cb0fcfb7`.

Integration preserves the `agentinfer.agentbench` package path, destination SPDX notices,
Node 20-compatible bridge tests, YAML package data, and bilingual user documentation.
Adjacent changes are limited to Replay CLI delegation, DSH identity headers, and Anthropic Replay sampling conversion.
The scheduler comparison launcher uses the destination controller's existing policy configuration.
