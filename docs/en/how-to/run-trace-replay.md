# Run Trace Replay

Replay reconstructs sessions, agents, dependencies, intervals, and token targets from a request trace, sends requests
to the backend, and records performance evidence. It does not launch the original agent, execute tools, or evaluate
SWE-bench correctness. Synthetic prompts do not establish exact text or performance parity with the original run.

## TraceLab replay

Each input line describes one LLM round with `provider`, `session_id`, `round_index`,
`input_tokens_total`, `prefix_tokens`, `newly_append_tokens`, `output_tokens`, `timing_events`, and `tools`.
The converter validates token splits, groups rounds by provider and session, and emits ordered IR v3 requests.
Tool metadata is retained for auditing; source tools are not executed.

```bash
vllm bench serve --agentinfer replay \
  --config agentinfer/agentbench/configs/replay_benchmark.yaml \
  --trace-type tracelab --trace-path /path/to/round_trace.jsonl \
  --prompt-shape tracelab_synthetic --endpoint /v1/chat/completions \
  --base-url http://127.0.0.1:8000 --model MODEL_NAME \
  --task-num 8 --max-concurrency 4 \
  --result-dir results/replay-tracelab-8x4
```

Keep `max_input_tokens` and `max_output_tokens` null for full token targets.
`prompt_calibration_tolerance_tokens` must be zero. Both `trace` and `lognormal` intervals are supported.
Task count and concurrency refer to runtime sessions; repeated samples have distinct identities and private content.

Each round sends one synthetic user message. The backend tokenizer probes template boundaries.
The plan separates input inheritance (`input_after`) from sending dependencies (`send_after`), without appending
live assistant responses. Rounds execute serially within a session; successors are released after predecessor
completion plus the configured interval. Root requests have no input predecessor; subsequent prefix reuse is
bounded by source cache targets, template boundaries, and input lengths.
Execution checks exact full-prompt token counts and the longest common prefix (LCP).
Missing SSE `[DONE]`, missing usage, or input/output token mismatches fail the request.
Source prefix counts need not equal observed vLLM cache hits; missing cache usage remains unavailable.

IR v3 uses `requests.jsonl` and `manifest.json` without empty text sidecars. Explicit analysis lives in
`unified_trace_ir.py`; the existing analyzer and Inferact IR v2 remain supported.
Token recipe profiles use Plan schema v2 / Planner v10; other paths retain v1 / v9.

Migrate old prompt shapes `claude_code_minimal_v1`, `trace_record`, and `token_recipe` to
`agentinfer_synthetic`, `inferact_synthetic`, and `tracelab_synthetic`, respectively. Old aliases are rejected.
The shape affects the workload fingerprint, so migrated configurations have different workload identities.
IR `prompt_source.kind=token_recipe`, `token_recipe_profile`, and `token_recipe_lcp_*` metric names are unchanged.
Inferact still preserves source human text and live assistant history; calibration residuals remain diagnostics
in `trace-record-validation.json`, without rewriting that text or terminating the session for those residuals.

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
| `replay.trace_type: agentinfer` | AgentInfer `requests.jsonl` input with the default `prompt_shape: agentinfer_synthetic`. |
| `replay.trace_type: inferact_codex_swebenchpro` | Raw Inferact JSON; requires `prompt_shape: inferact_synthetic`, `interval_mode: lognormal`, and `/v1/chat/completions`. |
| `replay.trace_type: tracelab` | Normalized, uncompressed JSONL; requires `prompt_shape: tracelab_synthetic`, zero calibration tolerance, and `/v1/chat/completions`. |
| `replay.trace_type: agentX` | Reserved with `prompt_shape: agentX_synthetic`; execution raises `NotImplementedError`. |
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
