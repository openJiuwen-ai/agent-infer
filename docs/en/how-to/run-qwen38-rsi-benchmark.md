# Qwen3.8-27B B300 RSI benchmark knowledge base

This page is the append-only working record for the 4×B300 serving target from
PR #25. It records the workload contract, measured iterations, rejected
hypotheses, and the next experiment. A measurement is not an acceptance
decision until the raw evidence and the independent quality gate are present.

## Goal and non-goals

The goal is the highest reproducible end-to-end output-token throughput for
`Qwen/Qwen3.8-27B` on four B300 GPUs, using the AgentBench
`inferact_codex_swebenchpro` trace replay as the performance workload and
GSM8K served through vLLM as the quality gate.

The replay reconstructs request timing, context and token targets. It does not
run the original Codex agent, execute its tools, or establish SWE-bench
correctness. The trace is therefore a serving workload, not an agent-quality
score. The raw trace is kept outside the repository because the 610-record
source is large; its URL, SHA256 and conversion manifest are recorded in each
experiment ledger entry.

The structured serving, runtime, kernel, evidence and promotion rules are kept
in the [Qwen3.8-27B B300 knowledge base](../../rsi/qwen38-b300-knowledge-base.md).
It incorporates the [vLLM kernel benchmark workflow](https://github.com/vllm-project/vllm/blob/main/.agents/skills/kernel-microbenchmark/SKILL.md),
the [vLLM Triton guidance](https://github.com/vllm-project/vllm/blob/main/.agents/skills/triton-kernel-writing/SKILL.md),
the [Z.ai dense-feedback account](https://z.ai/blog/glm-built-its-inference-infrastructure)
and the [NVlabs KDA workflow](https://github.com/NVlabs/kda).

## Frozen controls

| Control | Value |
| --- | --- |
| Hardware | NVIDIA B300 SXM6 AC, four GPUs, NVLink |
| Model | `Qwen/Qwen3.8-27B` |
| Model weight format | BF16 safetensors, local snapshot |
| vLLM | 0.29.0 |
| API | OpenAI-compatible `/v1/chat/completions` |
| Trace type | `inferact_codex_swebenchpro` |
| Trace source | `Inferact/codex_swebenchpro_traces`, 610 records |
| Trace prompt mode | `trace_record`, exact prompt calibration |
| Accuracy source | OpenAI GSM8K `test` JSONL |
| Accuracy protocol | deterministic request seed, `temperature=0`, `top_p=1`, `top_k=0`, explicit `####` extraction |

The current serving environment also exposes an editable vLLM-Omni checkout
with a vLLM 0.28/0.29 version warning. Direct `vllm serve` was used without
the Omni mode, but this remains a validity limitation until a clean vLLM-only
environment is measured. The warning and the exact environment path must stay
in every result manifest.

## Iteration log

The frozen trace identifiers for this run are:

* raw trace SHA256: `670f1ae8325fd70aac6ae6bdf4b03bbbd740d6ac8f2b49d8a02daaee3193fbc3`
* cached Unified IR bundle SHA256: `48dce67812a7138cc97e9caecd2a166430e32d047e2e6eb7d7dda94972716d0f`
* converted request IR SHA256: `d4d1910747438bd32673757f6d67af4544cd2cc652dc6e0fbddeb4ea35c1614f`
* GSM8K test SHA256: `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`, 1,319 rows

### I0: make the target path executable

Change: connected the PR #25 RSI evidence plane to a real vLLM TP4 endpoint,
reused the SHA-bound converted trace, enabled prefix caching and the 262,144
token context limit, and ran one fixed 14-request smoke.

Effect: 14/14 requests passed exact input accounting and returned successfully.
The run produced 849 output tokens at 79.045 output tok/s, 35,193.178 input
tok/s, 1.303 requests/s, 81.576 ms TTFT p50 and a 0.893926 prefix-cache hit
rate. This was a feasibility smoke with a 64-token output cap, so it is not a
throughput winner.

### I1: reject the first full-sample attempt

Change: increased the fixed sample to eight sessions and concurrency eight while
requiring zero prompt-calibration residual.

Effect: the run is recorded as `failed`, not as a performance result. Three of
eight tasks completed; five failed before send at a one-token current-turn
calibration boundary. The backend returned 97/97 attempted requests, while 154
dependent requests were skipped. Exact residuals among attempted requests were
zero, but coverage was incomplete and all throughput fields remain null.

### I2: deterministic boundary repair and valid TP4 sample

Change: added literal punctuation boundary candidates to suffix repair and ran
seed 228 with two sessions at concurrency two, keeping TP4, prefix caching,
model, context and exact accounting fixed.

Effect: 36/36 requests passed with zero failures, zero skips and zero maximum
calibration residual. The run reached 203.280 output tok/s, 29,167.320 input
tok/s, 0.619 requests/s, 174.720 ms TTFT p50 and a 0.942330 prefix-cache hit
rate. This is the current best valid completed comparison, but it covers two
sampled sessions and still needs repeated confirmation before being treated as
a final capacity claim.

### I3: compare two TP2 replicas

Change: split the same four GPUs into two TP2 vLLM instances behind a sticky
local proxy, using the same seed, two-session sample and exact calibration.

Effect: 36/36 requests passed with zero failures and zero skips, but aggregate
output throughput was 154.174 tok/s, 24.16% below I2. Input throughput was
22,121.459 tok/s and the backend prefix-cache hit rate was 0.928004. Request
TTFT is deliberately null because the proxy used for this comparison did not
preserve streaming timing semantics. TP4 remains the serving candidate among
the completed valid comparisons.

### I4: independent GSM8K quality gate

Change: sent all 1,319 official GSM8K test questions to the TP4 endpoint with
concurrency 16, `temperature=0`, `top_p=1`, `top_k=0`, seed 42 and a 1,024
token output cap; only the final `####` answer was parsed.

Effect: 1,254/1,319 answers were correct, for 0.950720 accuracy (95.07%).
There were 1,276 parsed predictions and zero transport errors; mean latency was
1.847 s and p95 latency was 6.558 s. This is a quality gate, separate from the
trace replay throughput measurement.

The RSI taxonomy, feedback evaluation, demo state machine, real append-only
ledger and read-only dashboard were all run. The real dashboard reads only
`experiments.jsonl`; HTTP POST is rejected, and new evidence is appended with
`python -m agentinfer.rsi experiments append` so every round keeps its
hypothesis, change, metrics, failure reason, next test and evidence hashes.

## Acceptance gates

An iteration can be called measured only when the result directory contains the
raw request trace, source analysis, replay plan, replay execution, summary,
Prometheus snapshots, server launch log, environment metadata and the exact
command. A candidate is eligible for promotion only when all replay requests
pass exact input accounting, the service is healthy, and the frozen GSM8K
protocol has a recorded accuracy result. Missing quality or environment data
is `inconclusive`, not a zero.

## Current result and 50-round effect

The corrected sweep is I11–I60: ten TP4 serving profiles with five measured
repetitions each, one unranked warmup per profile, and the same seed-228,
two-task, concurrency-two replay contract. I5–I9 remain visible as command
harness failures from the first attempt; I10 verified the repaired AgentInfer
dispatcher before the 50 valid rounds started.

The best quality-qualified profile is BF16 TP4 with prefix caching and
`--max-num-batched-tokens=32768`: 210.385 median output tok/s over five
replay-valid runs, +0.98% versus the repeated baseline median of 208.347, and
1255/1319 = 95.1478% GSM8K accuracy. FP8 KV had a slightly higher replay median
(210.658 tok/s) but scored 1251/1319 = 94.8446%, below the BF16 95.0720%
reference, so it was rejected. The `batch-8192` profile produced one fast
complete run but failed coverage in its other four repetitions.

A post-sweep five-run interaction check combining `batch-32768` with
`async-on` was replay-valid but reached only 208.134 median tok/s; the controls
remain separate. The complete per-round optimization point and effect is in
[qwen38-b300-50-round-results.md](../../rsi/qwen38-b300-50-round-results.md).
The review assets are [the dashboard](../../assets/rsi/qwen38-rsi-dashboard.png)
and [the updated architecture](../../assets/rsi/qwen38-b300-architecture.png).
The layered knowledge base records the external vLLM, Z.ai and NVlabs KDA
workflow rules and the promotion decisions.
