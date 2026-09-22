# Run Trace Replay

Replay reconstructs sessions, agents, dependencies, intervals, and token targets from a request trace, sends requests
to the backend, and records performance evidence. It does not launch the original agent, execute tools, or evaluate
SWE-bench correctness. Synthetic prompts do not establish exact text or performance parity with the original run.

## TraceLab replay

Each input line describes one LLM round with `provider`, `session_id`, `round_index`,
`input_tokens_total`, `prefix_tokens`, `newly_append_tokens`, `output_tokens`, `timing_events`, and `tools`.
The converter preserves source token counts, groups rounds by provider and session, and emits ordered token-recipe IR requests.
Input and output targets come from their total counts; prefix and appended counts are retained without checking their sum.
Tool metadata is retained for auditing; source tools are not executed.

Copy the example YAML to `/path/to/tracelab.yaml` and set `replay.prompt_calibration_tolerance_tokens: 0`
before applying these CLI overrides:

```bash
vllm bench serve --agentinfer replay \
  --config /path/to/tracelab.yaml \
  --trace-type tracelab --trace-path /path/to/round_trace.jsonl \
  --endpoint /v1/chat/completions \
  --base-url http://127.0.0.1:8000 --model MODEL_NAME \
  --task-num 8 --max-concurrency 4 \
  --result-dir results/replay-tracelab-8x4
```

Replay uses the full input and output token targets from the trace.
`prompt_calibration_tolerance_tokens` must be zero. Both `trace` and `lognormal` intervals are supported.
Task count and concurrency refer to runtime sessions; repeated samples have distinct identities and private content.

The first round sends a synthetic user message. Each continuation uses `context_after` to retain the previous
messages, append the live assistant response, and add a new synthetic user message. `send_after` still controls
release after predecessor completion plus the configured interval. The shared graph executor requires a successful
predecessor response; failed requests or prompt construction failures cause context-dependent successors to skip.

The backend tokenizer counts the full conversation with its chat template. Calibration changes synthetic user text
to meet the input target; it does not truncate live assistant responses. Strict mode fails when history alone exceeds
the target. Adaptive mode may trim a bounded amount of historical synthetic user text, but TraceLab never resets
or silently discards the assistant history to fit the target. An unreachable target fails the request.
Missing SSE `[DONE]`, missing usage, or input/output token mismatches also fail the request.

Source `prefix_tokens` remain audit evidence, not an exact LCP target: retaining live conversation history can produce
a different prefix from the source trace. Fixed-prefix template profiles and exact-LCP metrics are no longer emitted.
Missing backend cache usage remains unavailable; first/continuation cache groups now follow `context_after`.

TraceLab IR uses `requests.jsonl` and `manifest.json` without text sidecars. Explicit analysis lives in
`unified_trace_ir.py`; the existing analyzer and Inferact text IR remain supported. Regenerate earlier TraceLab IR and
plans: `context_after` replaces `input_after`. Converter, planner, sampler, and timing identifiers no longer carry
version suffixes; `schema_version` remains part of format validation. Identifier and hash-material changes affect
workload fingerprints, runtime identities, and seeds. Historical prefix-copy runs are not equivalent workloads.

Prompt shape is derived automatically from `replay.trace_type`. Remove `replay.prompt_shape` from existing YAML
files and remove `--prompt-shape` from commands; explicit prompt-shape configuration is rejected.
Removing the field from serialized configuration changes workload fingerprints and runtime identities.
IR `prompt_source.kind=token_recipe` is unchanged; its token targets now describe synthetic user turns with live
assistant context. Changes to plans or prompt construction affect workload fingerprints and runtime identities.
Inferact still preserves source human text and live assistant history; calibration residuals remain diagnostics
in `trace-record-validation.json`, without rewriting that text or terminating the session for those residuals.

## Prepare the backend

Install the project and start vLLM with the target model. The backend must expose `/tokenize`, `/detokenize`,
`/metrics`,
and the configured inference endpoint. When using a Router, set `backend.tokenizer_base_url` to the tokenizer service
and `backend.metrics_url` to the complete vLLM metrics URL. For `/v1/messages`, launch vLLM with
`agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware` to propagate identity and sampling parameters.
See [vLLM integration](integrate-vllm.md) for deployment options.

Inferact conversion discovers the Backend tokenizer through `/v1/models` and the optional `/tokenizer_info`.
When matching tokenizer files are available on the same host and raw-text and chat-template token-ID probes match
the Backend exactly, conversion and Replay calibration use the local tokenizer. Conversion may use incremental
counting only for the probed message counts (1, 3, 5, and 9) when additivity probes pass; other lengths and runtime
calibration always count the full chat template. Unavailable or incompatible local tokenizers fall back to
`/tokenize`. vLLM exposes
`/tokenizer_info` only when started with `--enable-tokenizer-info-endpoint`; enable it when the server overrides the
model's chat template.

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
| `replay.trace_type: agentinfer` | AgentInfer `requests.jsonl` input; automatically selects `agentinfer_synthetic`. |
| `replay.trace_type: inferact_codex_swebenchpro` | Raw Inferact JSON; automatically selects `inferact_synthetic`. Requires `interval_mode: lognormal`, zero calibration tolerance, and `/v1/chat/completions`. |
| `replay.trace_type: tracelab` | Normalized, uncompressed JSONL; automatically selects `tracelab_synthetic`. Requires zero calibration tolerance and `/v1/chat/completions`. |
| `replay.trace_type: agentX` | Automatically selects `agentX_synthetic`; reserved, execution raises `NotImplementedError`. |
| `replay.interval_mode` | `trace` preserves historical intervals; `lognormal` generates configured intervals. |
| `replay.sample_seed` | Reproducible session selection and backend sampling seed. |
| `replay.context_adjustment_mode` | `strict` rejects non-append-only context; `adaptive` audits trims and context resets. |
| `replay.request_timeout_seconds` | Per-request timeout. |

For all fields, defaults, and constraints see the [Replay configuration
model](../../../agentinfer/agentbench/replay/config.py)
and [example YAML](../../../agentinfer/agentbench/configs/replay_benchmark.yaml).
Run `vllm bench serve --agentinfer replay --help` to list supported CLI overrides.

### Align input lengths while retaining live answers

Inferact `inferact_synthetic` freezes sent messages, including old filler, and retains live assistant content and its
separate `reasoning_content`. It pads the newest user text when below target, or trims only that text's suffix
when over target. Trimming uses character boundaries; every candidate is counted with the full chat template.
The historical trimming/reset rules of `context_adjustment_mode` do not apply to this path.

Inferact always requires exact per-request input lengths, with no additional mode or tolerance configuration.
Nonzero `prompt_calibration_tolerance_tokens` values are rejected when loading Inferact configuration. Remove a legacy
nonzero setting or explicitly set it to zero; synthetic prompts retain their configurable tolerance.

If no current-user prefix fits the target, bounded suffix repair cannot reach the exact target,
or calibration changes the token prefix shared by the original and empty-user templates, the request fails before
inference. Missing `usage.prompt_tokens` or a backend input count differing from the target also fails the request;
dependent requests are skipped. History is never trimmed to force a fit.

Per-request calibration in `replay-execution.json` records `trimmed_current_user_tokens`,
`trimmed_current_user_characters`, `preserved_prefix_tokens`, and `backend_input_residual_tokens`.
Source-text trimming is separate from `trimmed_filler_tokens`. In `trace-record-validation.json`,
`input_length_comparable` is true only when all planned requests succeed and backend input usage exactly matches targets.
This checks input lengths; it does not guarantee identical Prefix Cache hits or preserve the meaning of trimmed text.

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
