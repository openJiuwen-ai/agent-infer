# AgentBench

AgentBench separates orchestration (`benchkit/`), Claude Code execution (`agents/`), and transparent Anthropic
request observation (`request_proxy/`). Baseline and Scheduler candidate runs both target vLLM through the same
Request Proxy.

Start with [Run a benchmark](../../docs/en/how-to/run-benchmark.md). Module boundaries and invariants are documented
under the [benchmark subsystem design](../../design/module/benchmarking/index.md).
