# AgentBench

AgentBench separates orchestration (`benchkit/`), Claude Code execution (`agents/`), and transparent Anthropic
request observation (`request_proxy/`). Baseline and Scheduler candidate runs both target vLLM through the same
Request Proxy.

Start with the [Benchmark Quickstart](../../docs/benchmarks/quickstart.md). Module boundaries and invariants are
documented under [`docs/design/module/benchmarking`](../../docs/design/module/benchmarking/index.md).
