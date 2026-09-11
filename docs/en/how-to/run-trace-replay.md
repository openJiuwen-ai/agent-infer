# Run Trace Replay

Replay reconstructs sessions, agents, dependencies, intervals, and token targets from a request trace, sends requests
to the backend, and records performance evidence. It does not launch the original agent, execute tools, or evaluate
SWE-bench correctness. Synthetic prompts do not establish exact text or performance parity with the original run.

## Prepare the backend

Install the project and start vLLM with the target model. The backend must expose `/tokenize`, `/detokenize`,
`/metrics`,
and the configured inference endpoint. When using a Router, set `backend.tokenizer_base_url` to the tokenizer service
and `backend.metrics_url` to the complete vLLM metrics URL. For `/v1/messages`, launch vLLM with
`agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware` to propagate identity and sampling parameters.
See [vLLM integration](integrate-vllm.md) for deployment options.

## Replay the bundled sample

Run from the repository root, replacing `MODEL_NAME` with the actual served model name:

```bash
vllm bench serve --agentinfer replay \
  --config agentinfer/agentbench/configs/replay_benchmark.yaml \
  --trace-path tests/agentbench/replay/claude_trace_8session_requests.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model MODEL_NAME --task-num 1 \
  --result-dir results/replay-smoke
```

The result directory must not exist. CLI paths resolve from the current directory; YAML paths resolve from the
configuration directory. The example includes synthetic prefix budgets for the sample; recalibrate these for other
recording sources. The backend context window must accommodate the request targets.

## Inputs and configuration

| Configuration | Meaning |
| --- | --- |
| `replay.trace_type: agentinfer` | AgentInfer `requests.jsonl` input with synthetic prompts. |
| `replay.trace_type: inferact_codex_swebenchpro` | Raw Inferact JSON; requires `prompt_shape: trace_record`, `interval_mode: lognormal`, and `/v1/chat/completions`. |
| `replay.trace_type: agentX` or `tracelab` | Reserved values that raise `NotImplementedError` during execution. |
| `replay.interval_mode` | `trace` preserves historical intervals; `lognormal` generates configured intervals. |
| `replay.sample_seed` | Reproducible session selection and backend sampling seed. |
| `replay.max_input_tokens` / `max_output_tokens` | Optional token caps; `null` preserves trace targets. |
| `replay.context_adjustment_mode` | `strict` rejects non-append-only context; `adaptive` audits trims and context resets. |
| `replay.request_timeout_seconds` | Per-request timeout. |

For all fields, defaults, and constraints see the [Replay configuration
model](../../../agentinfer/agentbench/replay/config.py)
and [example YAML](../../../agentinfer/agentbench/configs/replay_benchmark.yaml).
Run `vllm bench serve --agentinfer replay --help` to list supported CLI overrides.

## Inspect and compare results

Inspect the status in `manifest.json` and the `summary.json`. Retain `requests.jsonl`, `replay-source-analysis.json`,
`replay-plan.json`, `replay-execution.json`, and `evidence/`. On failure inspect `replay-error.json`; partial runs may
not contain every artifact. Inferact conversion artifacts are stored under `convert_result/` in the result directory.

```bash
vllm bench serve --agentinfer compare \
  --baseline results/replay-baseline --candidate results/replay-candidate
```

Fair comparisons use the same trace, seed, prefix budgets, concurrency, and deployment settings. Restart the service
before each cold run and verify starting Prefix Cache metrics. Replay provides no task correctness conclusion;
see [benchmark methodology](../explanation/benchmark-methodology.md).
