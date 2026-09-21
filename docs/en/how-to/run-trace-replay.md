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

### Align input lengths while retaining live answers

Inferact `trace_record` freezes sent messages, including old filler, and retains live assistant content and its
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
