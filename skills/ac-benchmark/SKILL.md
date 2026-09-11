---
name: ac-benchmark
description: Use when measuring AgentInfer cache performance against a target engine (vLLM or vLLM Prefix Cache); discovers current benchmark paths before running, captures artifacts, and compares against baseline.
allowed-tools:
  - Bash
  - Read
  - Glob
  - Grep
---

# ac-benchmark

Run the AgentInfer benchmark harness against a target LLM serving engine using
the current config-driven framework, capture reproducible artifacts, and compare
against a cold baseline.

## WHEN TO INVOKE

- The user asks to "benchmark AgentInfer", "measure cache hit rate",
  "compare latency/throughput vs vLLM", or "run the benchmark suite".
- A change to cache or scheduling behavior needs performance evidence before merge.

Do NOT invoke for: unit-testing a function, profiling unrelated code, or comparing two non-baseline runs against each other.

## STEPS

1. **Discover the benchmark surfaces from the current checkout**:

   - Configs and datasets: `agentinfer/agentbench/`
   - Focused tests and launch helpers: `tests/agentbench/`
   - CLI entry point: `vllm bench serve --agentinfer`
   - Runbook: `docs/en/how-to/run-benchmark.md`

   Read the runbook and CLI help before constructing commands. Do not reuse
   paths or flags from an older branch.

2. **Define the full experiment matrix before launching**. Record every intended
   mode, workload shape, task selection, agent profile, concurrency, and trial.
   Explicitly list any omitted shape or mode. Baseline and candidate must use the
   same commit, model, ordered task list, profile, concurrency, timeouts, tensor
   parallelism, vLLM options, and hardware.

3. **Prepare inputs and preflight the environment**. Derive the data directory
   from the selected config/runbook. For the current SWE-bench setup, the command
   is:

   ```bash
   vllm bench serve --agentinfer prepare swebench \
     --output-dir agentinfer/agentbench/data/swebench
   ```

   Confirm the supported vLLM version declared by the current checkout (currently
   `0.23.0`), branch/commit, host alias, working directory, virtualenv, service
   port, dataset/config paths, result directories, session/process names, stop
   conditions, and exact launch commands. Result directories must not already
   exist.

4. **Run each arm from cold state**. Start the serving backend explicitly, verify
   health, then run the same workload for both arms. Derive task count,
   concurrency, agent executable, and profile from the experiment matrix and
   config. For example:

   ```bash
   vllm bench serve --agentinfer run \
     --config <config.yaml> \
     --task-num <task-count> \
     --max-concurrency <concurrency> \
     --agent-executable <agent-command> \
     --agent-profile <profile> \
     --result-dir <result-dir>
   ```

   The current harness supports `single` and `plan-subagent`; the latter requires
   an approved Claude Code permission setup. The CLI does not start vLLM. Fully
   stop the baseline service before starting the candidate so Prefix Cache state
   and cumulative metrics reset. Retain both service logs. Use repeated cold runs
   for release decisions.

5. **Summarize and compare source artifacts directly**:

   ```bash
   vllm bench serve --agentinfer summarize results/agentinfer/run1 \
     --output-csv combined-summary.csv

   vllm bench serve --agentinfer compare \
     --baseline results/vllm/run1 \
     --candidate results/agentinfer/run1
   ```

   For repeated trials, pass every run directory for each side. Parse
   `manifest.json`, `summary.json`, `requests.jsonl`, and
   `evidence/vllm_metrics_*.prom`; do not manually transcribe report values.

6. **Report performance and evidence quality**. Cover at minimum end-to-end wall
   time, prefix hit rate, task execution status, TTFT percentiles, source-health
   evidence, cold-state confirmation, and unavailable artifacts. `completed`
   means agent execution completed; it is not SWE-bench correctness. Report
   resolved/unresolved only after separate SWE-bench-compatible evaluation.

7. **Report regressions and omissions explicitly**. Do not silently replace a
   baseline, conceal a worse metric, or present a directional single trial as a
   release conclusion.

## DON'T

- Don't compare against a non-baseline run as if it were the baseline.
- Don't launch until the full mode/shape matrix and omissions are recorded.
- Don't compare warm and cold arms; fully restart vLLM between them.
- Don't delete old result directories or service logs — they are evidence.
- Don't manually transcribe metrics that can be parsed from artifacts.
- Don't equate `completed` tasks with SWE-bench `resolved` tasks.
- Don't claim performance improvement without backing data.

## AFTER

Once metrics are captured and compared, invoke `ac-review` to validate any
harness or baseline changes before opening the PR.
