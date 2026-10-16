# Run Trace Replay

Replay reconstructs sessions, agents, dependencies, intervals, and token targets from a request trace, sends requests
to the backend, and records performance evidence. It does not launch the original agent, execute tools, or evaluate
SWE-bench correctness. Synthetic prompts do not establish exact text or performance parity with the original run.

## TraceLab replay

Each input line describes one LLM round with `provider`, `session_id`, `round_index`,
`input_tokens_total`, `prefix_tokens`, `newly_append_tokens`, `output_tokens`, `timing_events`, and `tools`.
The converter preserves source token counts, groups rounds by provider and session, and emits ordered token-recipe IR requests.
Within each session, `round_index` must start at 0 and increase contiguously; missing rounds fail conversion.
Input and output targets come from their total counts; prefix and appended counts are retained without checking their sum.
Tool metadata is retained for auditing; source tools are not executed.

For a quick start, omit `--config`; Replay selects the packaged TraceLab template from `--trace-type`:

```bash
vllm bench serve --agentinfer replay \
  --trace-type tracelab \
  --base-url http://127.0.0.1:8000 --model MODEL_NAME \
  --task-num 8 --max-concurrency 4 \
  --result-dir results/replay-tracelab-8x4
```

Without `--config`, `--trace-type`, `--task-num`, `--base-url`, and `--model` are required; `--max-concurrency`
defaults to `1`. `task_num` is required for every Replay type and must be a positive integer.
When TraceLab has `trace_path: null`, Replay downloads the pinned `v0.0.2` snapshot of `UW-SyFI/TraceLab` and
extracts the first `task_num` complete sessions in first-seen source order. Every round belonging to those sessions is
retained. The uncompressed JSONL is cached at
`~/.cache/agentinfer/datasets/tracelab/v0.0.2/first-<task_num>-sessions.jsonl`, or below `XDG_CACHE_HOME` when set,
and reused by later runs. If fewer source sessions exist, the planner cycles them deterministically to `task_num`.
An explicit `--trace-path` always wins
and disables download. With `--config`, all CLI overrides are optional and explicitly supplied overrides take precedence.
YAML paths resolve relative to that file; CLI paths resolve relative to the current directory.

For Inferact with `trace_path: null`, Replay streams the pinned JSON array and closes the response after
`task_num` complete records. Network chunks may read slightly ahead; the unread suffix is not validated.
The subset is written atomically to
`~/.cache/agentinfer/datasets/inferact_codex_swebenchpro/<revision>/first-<task_num>-records.json`
(respecting `XDG_CACHE_HOME`). Existing subsets and complete Hugging Face cache files are reused without
downloading. `HF_HUB_OFFLINE=1` fails explicitly when neither cache is available.

| `--trace-type` | Packaged template |
| --- | --- |
| `agentinfer` | `replay_agentinfer.yaml` |
| `inferact_codex_swebenchpro` | `replay_inferact.yaml` |
| `tracelab` | `replay_tracelab.yaml` |
| `agentX` | `replay_agentX.yaml` |

Replay uses the trace's input and output token targets. TraceLab conversion and planning preserve `output_tokens: 0`.
The current backend cannot execute zero output tokens exactly, so preparation fails explicitly instead of changing
that target to one. AgentX retains its separate zero-output planning rule described below.
`prompt_calibration_tolerance_tokens` must be zero. Both `trace` and `lognormal` intervals are supported.
Task count and concurrency refer to runtime sessions; repeated samples have distinct identities and private content.

A single warmup request uses `max(prefix_tokens at round_index=0 across all converted sessions) + 1` input tokens,
regardless of which sessions the planner selects. The converter records this as `summary.warmup_input_tokens` in the
IR manifest. Each selected session's first round takes its requested prefix from that warmup input. Later rounds
take the first
`min(prefix_tokens, previous input length + actual previous output length)` token IDs from the prior request in the
same session, then add new tokens to reach `input_tokens_total`. Source and effective prefix lengths and any
shortfall are recorded. Warmup usage and time are excluded from measured Replay results.

TraceLab sends token ID arrays to `/v1/completions`; the backend must return generated token IDs when
`return_token_ids` is requested. The configured chat endpoint identifies the OpenAI backend, while
`chat_template_kwargs` do not apply to TraceLab token ID requests. The model context window must fit the warmup
and every input plus output target. Set `experiment.task_timeout_seconds: null` to avoid truncating long sessions;
the packaged TraceLab template does this by default.

`send_after` still controls release after predecessor completion plus the configured interval. The shared graph
executor requires a successful predecessor response; failed requests cause dependent successors to skip. Missing
SSE `[DONE]`, missing usage or generated token IDs, and input/output token mismatches fail a request. An incomplete
TraceLab run is marked failed. Backend cache hits may still differ from the constructed prefix lengths.

TraceLab IR uses `requests.jsonl` and `manifest.json` without text sidecars. Explicit analysis lives in
`unified_trace_ir.py`; the existing analyzer and Inferact text IR remain supported. Regenerate earlier TraceLab IR and
plans: `context_after` replaces `input_after`. Converter, planner, sampler, and timing identifiers no longer carry
version suffixes; `schema_version` remains part of format validation. Identifier and hash-material changes affect
workload fingerprints, runtime identities, and seeds. Historical prefix-copy runs are not equivalent workloads.

Prompt shape is derived automatically from `replay.trace_type`. Remove `replay.prompt_shape` from existing YAML
files and remove `--prompt-shape` from commands; explicit prompt-shape configuration is rejected.
Removing the field from serialized configuration changes workload fingerprints and runtime identities.
IR `prompt_source.kind=token_recipe` is unchanged; its token targets drive the synthetic token prefix tree.
Changes to plans or prompt construction affect workload fingerprints and runtime identities.
Inferact preserves live assistant history and calibrates only the current user suffix. Backend usage mismatches
fail the request and its context-dependent successors; validation evidence remains in `trace-record-validation.json`.

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
  --base-url http://127.0.0.1:8000 \
  --model MODEL_NAME --task-num 8 --max-concurrency 4 \
  --result-dir results/replay-smoke
```

The result directory must not exist. CLI paths resolve from the current directory; YAML paths resolve from the
configuration directory. The example includes synthetic prefix budgets for the sample; recalibrate these for other
recording sources. The backend context window must accommodate the request targets.

## Inputs and configuration

| Configuration | Meaning |
| --- | --- |
| `replay.trace_type: agentinfer` | AgentInfer `requests.jsonl` input; a null `trace_path` uses the packaged eight-session dataset. Automatically selects `agentinfer_synthetic`. |
| `replay.trace_type: inferact_codex_swebenchpro` | Raw Inferact JSON; a null `trace_path` downloads a pinned revision and caches up to the first `task_num` records. Automatically selects `inferact_synthetic`. Requires `interval_mode: lognormal`, zero calibration tolerance, and `/v1/chat/completions`. |
| `replay.trace_type: tracelab` | Normalized, uncompressed JSONL; a null `trace_path` downloads and extracts the first `task_num` complete sessions. Automatically selects `tracelab_synthetic`. Requires zero calibration tolerance; uses token IDs through `/v1/completions` on the configured vLLM service. |
| `replay.trace_type: agentX` | Nested session JSONL with 64-token local hash blocks; uses complete token-ID snapshots and `/v1/completions`. Source requests with `out=0` generate one token during Replay. |
| `replay.interval_mode` | `trace` preserves historical intervals; `lognormal` generates configured intervals. |
| `replay.sample_seed` | Reproducible session selection and backend sampling seed. |
| `replay.context_adjustment_mode` | Controls generic synthetic prompt calibration; TraceLab uses the token prefix rule above. |
| `experiment.task_timeout_seconds` | Per-session timeout; `null` disables it. |
| `replay.request_timeout_seconds` | Per-request timeout. |

For all fields, defaults, and constraints see the [Replay configuration
model](../../../agentinfer/agentbench/replay/config.py)
and [example YAML](../../../agentinfer/agentbench/configs/replay_benchmark.yaml).
Run `vllm bench serve --agentinfer replay --help` to list supported CLI overrides.

### Replay AgentX hash snapshots

Set `replay.trace_type: agentX`, an explicit local `replay.trace_path`, and
`backend.endpoint: /v1/completions`. Each source line is a session. The converter
expands lead requests and the requests nested inside subagent groups, retaining
their shared session timeline. A request's `hash_ids` describes its complete input;
live output is measured but never appended to the next input. The configured
Backend model supplies all token IDs. Source model names remain in analysis and
partition generated blocks within a runtime session. Repeated samples have
different runtime session IDs and different token blocks.
AgentX requires `prompt_calibration_tolerance_tokens=0`. The default already supplies zero, so omit it from
the AgentX YAML; an explicit zero remains valid when serialized configurations are reloaded.

The source's `t` and `api_time` infer the most recently completed scheduling
predecessor. This is a timing relationship, not proof of parent/child causality;
subagent identity is retained while `blocks_parent` is false. The converter
keeps zero-output records for coverage; planning maps their output target to one
token for Backend execution and exact response validation. The source output count
remains zero in analysis. Set the Backend context limit high enough for the selected
request's `input_tokens + planned_output_tokens`. `replay.max_inflight_requests` bounds
simultaneous AgentX requests within each runtime session.
Snapshots are read and rendered after the planned release time and inside the session semaphore,
bounding the number of prepared large inputs. Completed requests do not retain snapshots or live answers
as continuation context. `scheduler_lag_seconds` includes semaphore wait and snapshot rendering, so
actual sends can occur after their planned release time; it is not an HTTP or Backend-only latency.

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

Replay marks a run `failed` if any planned task or request fails or is missing, while retaining available results.
The same comparison command accepts finalized failed runs and automatically displays failure status, request
coverage, and available node errors alongside descriptive metrics.
It does not treat an incomplete run as a successful performance sample. Add `--json` for all run diagnostics;
confidence intervals and significance are omitted (`null`) when any run failed. See
[CLI reference](../reference/benchmark-cli.md#compare) for artifact requirements.

Fair comparisons use the same trace, seed, prefix budgets, concurrency, and deployment settings. Restart the service
before each cold run and verify starting Prefix Cache metrics. Replay provides no task correctness conclusion;
see [benchmark methodology](../explanation/benchmark-methodology.md).

## Token-ID mode for scheduling and cache comparisons

For AgentInfer synthetic workloads, merge these fields into your YAML. Packaged templates retain text compatibility:

```yaml
backend:
  endpoint: /v1/chat/completions
replay:
  prompt_transport: completion_token_ids
  tokenizer_max_concurrency: 16
  prompt_calibration_tolerance_tokens: 0
  context_adjustment_mode: strict
```

Alternatively add `--endpoint /v1/chat/completions --prompt-transport completion_token_ids` to the existing command.

Token-ID mode requires `prompt_calibration_tolerance_tokens=0`. For TraceLab and AgentX this also applies to
`prompt_transport: auto`; nonzero values produce an error naming the trace type. AgentInfer text mode retains
its configurable input tolerance. Inferact continues to require zero tolerance in text mode.

Requests use `/v1/completions` on the same service. The backend and Router must support token-ID input,
streamed `return_token_ids`, and identity metadata. This preserves prefix scopes and token budgets, without text
Chat template semantics. The default `auto` retains AgentInfer/Inferact text and TraceLab/AgentX token-ID transport;
`chat_text` explicitly selects text. Inferact rejects token-ID mode; TraceLab and AgentX reject text mode.
AgentX also accepts explicit `completion_token_ids` with its existing `/v1/completions` configuration.

AgentInfer roots contain global, session, and request-private segments. Sibling Subagents share their session segment;
different sessions generate separate content. Continuations append actual output IDs and filler to previous input IDs,
without retokenizing full history. `strict` fails if the target cannot contain history and actual output.
With `adaptive`, the plan selects `trim` or `reset`. Trimming removes the ends of the nearest historical private filler
ranges, preserving shared, tool, extra system segments, and all actual output. It records `trimmed_filler_tokens` and
`preserved_prefix_tokens`, and fails if private filler is insufficient. Reset explicitly rebuilds a root prompt.
TraceLab retains its source `prefix_tokens` contract.

Global preparation validates the plan and deduplicates consumed global shared segments, skipping unused and zero-length
segments. Other prompt data is prepared only after a task acquires an `experiment.max_concurrency` slot. Its roots are
built before that session's release clock starts; completed, failed, or cancelled tasks release private tokenizer caches
and cached source text. Full prompts are retained by active task graphs, rather than all sampled tasks. This is a task
window, not a byte limit: one large session can still retain large histories. AgentX snapshots remain lazy, rendered
after release inside the session semaphore. Concurrent misses share one operation. Tokenize, detokenize, and count
requests share `tokenizer_max_concurrency`. Cancelling one waiter does not cancel shared work; task cleanup cancels
remaining private work, and shutdown cancels all remaining preparation. Global preparation uses `run_timeout_seconds`,
or `request_timeout_seconds` when unset. Task preparation uses the task timeout and measured execution's run timeout.

Text mode uses a backend-validated local tokenizer when available and retains full-template calibration, otherwise
falling back to backend counting. Fixed turn-overhead estimates are not accepted as exact counts. All modes require
protocol completion and matching input/output usage. Text retains its existing input tolerance; output must match
exactly. Messages input verification includes cache creation/read tokens. AgentInfer and TraceLab token-ID continuations
require output IDs with valid types and exact length. AgentX checks usage and stream completion without requiring output
IDs, because each input is a complete source snapshot. Missing or mismatched required evidence fails the request.

`evidence/replay_preparation.json` records global setup time and the active-task preparation policy separately.
Throughput measures execution after global setup and the initial metrics capture, excluding final collection.
It includes active task prompt preparation, so slow tokenization can reduce measured throughput. Do not subtract summed
prompt preparation times from wall time: preparation may overlap other tasks' inference. Per-request records include
preparation start/end offsets, `prompt_prepare_seconds`, `prompt_prepare_mode`, planned/actual send offsets,
and `scheduler_lag_seconds`. Preparation excludes context dependency wait; root offsets can be negative relative to
their session's release clock, even though root preparation is inside the measured window. Calibration records preserve
`source_output_tokens` separately from `effective_output_tokens` and `backend_output_tokens`: AgentX source output zero
remains visible as zero when its execution target is one. `backend_target_met` compares actual usage with execution targets.

Every adapter requires exact task, node, and persisted request identity coverage. Missing, duplicate, failed, or skipped
requests make the run `failed` and cause a failing CLI exit, while preserving execution and error evidence. Incomplete
runs cannot serve as successful performance comparison samples.
