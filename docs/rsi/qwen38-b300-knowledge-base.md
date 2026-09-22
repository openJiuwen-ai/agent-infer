# Qwen3.8-27B · 4×B300 RSI knowledge base

This is the working index for the serving optimization loop. It separates
portable engineering rules from claims that have been measured on this exact
model, vLLM version, workload and four-GPU host. A reference can suggest a
hypothesis; only a controlled run can promote it to a serving conclusion.

## Layered knowledge model

### System and serving architecture

This layer covers the request path, API streaming, AgentBench replay, vLLM
scheduler, KV cache, tensor/data parallel placement, GPU topology and process
readiness. Its primary evidence is the complete replay result, vLLM metrics,
server log, launch command and environment manifest.

### Runtime and scheduler behavior

This layer covers chunked prefill, scheduler token budgets, asynchronous
scheduling, sequence limits, streaming interval, prefix-cache reuse, CUDA
graphs, compilation and KV-cache dtype. Each change must modify one serving
control while trace SHA, sample seed, replay concurrency, calibration tolerance,
model, TP layout and endpoint remain fixed.

### Kernel and numerical behavior

This layer is relevant when profiling attributes a serving change to a kernel,
fusion, attention path or recurrent state update. It requires a correctness
oracle and representative shape coverage before an isolated kernel speedup can
be considered an E2E optimization. KDA material belongs here; it is not a
substitute for a Qwen3.8 model-level quality result.

### Experiment protocol and evidence

Every round records:

1. the question or hypothesis;
2. the single change and frozen controls;
3. correctness and coverage status;
4. steady-state E2E throughput and latency;
5. local evidence such as logs, profiles or microbenchmarks;
6. the observation, decision and next test.

The ledger is append-only. A failed or inconclusive round remains visible and
cannot become a zero or a performance win through missing fields. Candidate
promotion requires repeated valid replay coverage and the independent GSM8K
quality gate.

## Rules incorporated from external references

### vLLM agent skills

The [vLLM kernel-microbenchmark skill](https://github.com/vllm-project/vllm/blob/main/.agents/skills/kernel-microbenchmark/SKILL.md)
adds these rules to this workflow:

* check correctness before timing and keep tolerances explicit;
* isolate the timed operation and report GPU, dtype, shapes, command, commit
  and relevant environment variables;
* treat explanations as hypotheses until an ablation, trace, generated code or
  profiler artifact supports them;
* for multi-GPU work, report topology, world size, TP configuration and the
  global timing boundary; rank-local time is not distributed latency;
* warm up CUDA graphs and stabilize clocks or use enough repetitions to share
  clock, thermal and rank-skew effects.

The [vLLM Triton-kernel-writing skill](https://github.com/vllm-project/vllm/blob/main/.agents/skills/triton-kernel-writing/SKILL.md)
adds boundary-shape correctness, explicit accumulation dtypes, generated-code
inspection without treating generated code as proof, and shape sweeps covering
decode plus representative prefill. These rules become mandatory if a future
round changes a kernel or promotes a kernel-derived serving profile.

### Z.ai dense feedback and RSI loop

The [Z.ai inference-infrastructure account](https://z.ai/blog/glm-built-its-inference-infrastructure)
defines dense feedback as local, attributable and actionable feedback. The
working interpretation for this benchmark is:

* use the E2E result to decide whether a candidate is useful, then use runtime
  events, metrics or a microbenchmark to localize the cause;
* choose the next observation based on the current hypothesis instead of
  collecting every possible log on every round;
* keep correctness, system behavior and performance as separate feedback
  channels;
* preserve optimization skeletons with applicability conditions and validation
  evidence, rather than copying an optimization without its shape and hardware
  constraints;
* keep architecture, concurrency, numerical semantics and promotion decisions
  under human review.

The prior I0–I4 run already follows this split: replay supplies performance,
the prompt calibration gate supplies request correctness, vLLM metrics supply
runtime behavior, and GSM8K supplies model-level quality.

### NVlabs KDA workflow

The [NVlabs KDA repository](https://github.com/NVlabs/kda) contributes the
kernel-task workflow and evidence topology:

* begin with a task contract containing the objective, constraints, validation
  command and promotion criteria;
* work in an isolated workspace and make each iteration small;
* record candidates, benchmark/evaluation results, profiling evidence and the
  final promotion decision;
* keep a reproducible workspace with plans, runs, profiles, benchmark tables
  and candidate records.

For this serving task, KDA's kernel-specific correctness and profiling gates
map to the kernel layer above. The benchmark adapter must keep the vLLM
baseline and candidate ABI and workload identical, and it must include wrapper,
dispatch and synchronization cost when the claim is E2E. The KDA repository's
kernel workflow is therefore a source of procedure and evidence discipline,
not evidence that a KDA kernel is used by Qwen3.8.

## Promotion ladder

| Stage | Required evidence | Permitted conclusion |
| --- | --- | --- |
| Feasibility | Healthy service and a successful request | The path runs |
| Replay-valid | All planned requests, exact input accounting, no skipped dependencies | The profile is benchmarkable |
| E2E candidate | Repeated replay-valid runs, fixed controls, median and spread | The profile is a throughput candidate |
| Quality-qualified | Candidate plus full deterministic GSM8K | The profile may be promoted for this model/task |
| Kernel-qualified | Kernel correctness oracle, boundary shapes, profiler/microbenchmark evidence and E2E confirmation | A kernel-derived optimization may be retained |

## Knowledge-base layout

The working set now has one stable index and generated evidence extracts:

```text
docs/rsi/qwen38-b300-knowledge-base.md   # controls, rules and promotion state
docs/rsi/qwen38-b300-50-round-results.md # per-round optimization point/effect
docs/rsi/qwen38-b300-architecture.mmd    # source architecture
docs/assets/rsi/qwen38-b300-architecture.png
docs/assets/rsi/qwen38-rsi-dashboard.png # review-friendly dashboard snapshot
tools/rsi_replay_sweep.py                # resumable controlled sweep
tools/render_rsi_dashboard.py            # deterministic PNG renderer
```

The append-only ledger remains the source of truth. The markdown result extract
and PNG are review artifacts generated from that ledger; they do not replace
the raw replay directories, Prometheus snapshots or evidence hashes.

## 50-round sweep result

The corrected sweep is I11–I60: ten profiles, five measured repetitions per
profile, one unranked warmup per profile. It keeps seed 228, two tasks,
concurrency two, exact calibration, the cached conversion bundle, TP4 on GPUs
0–3, BF16 weights, prefix caching and max context 262144 fixed. I5–I9 are
preserved command-harness failures from the first attempt, when the environment
used the upstream vLLM console script instead of the AgentInfer dispatcher.
The runner now invokes the repository dispatcher explicitly and I10 verified
the corrected path before the sweep.

| Profile | Replay-valid | Median output tok/s | Range | Delta vs repeated baseline | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline-current | 5/5 | 208.347 | 205.072–212.562 | +0.00% | control |
| no-prefix-cache | 5/5 | 143.964 | 143.515–146.455 | -30.90% | reject |
| batch-8192 | 1/5 | 210.503* | 210.503 | +1.03%* | reject coverage |
| batch-32768 | 5/5 | 210.385 | 202.529–213.529 | +0.98% | quality-qualified |
| batch-65536 | 5/5 | 208.680 | 201.952–212.749 | +0.16% | no promotion |
| async-on | 5/5 | 209.065 | 206.643–211.929 | +0.34% | no promotion |
| async-off | 5/5 | 186.674 | 182.972–188.543 | -10.40% | reject |
| stream-8 | 5/5 | 208.524 | 203.478–212.204 | +0.08% | no promotion |
| max-seqs-8 | 5/5 | 206.784 | 204.168–211.157 | -0.75% | no promotion |
| kv-fp8 | 5/5 | 210.658 | 203.665–213.419 | +1.11% | reject quality |

`batch-8192` produced one complete 36/36 replay at 210.503 tok/s, but its
other four repetitions completed only 27/36 requests and skipped eight
dependent requests. The single high result is excluded from promotion.

The FP8 KV profile was the fastest median, but its independent GSM8K result was
1251/1319 = 94.8446%, below the BF16 reference 1254/1319 = 95.0720%, so it is
rejected. The BF16 `batch-32768` candidate scored 1255/1319 = 95.1478% and
passes the quality gate. A post-sweep interaction check I63–I67 combined
`batch-32768` with `async-on`; it was replay-valid but its median was 208.134
tok/s, below the single-control candidate, so the controls are not combined.

The current quality-qualified choice is TP4 with prefix caching, BF16 weights
and `--max-num-batched-tokens=32768`: 210.385 median output tok/s on this
36-request replay sample and 95.1478% GSM8K accuracy. This is the best observed
candidate under the frozen contract, rather than a claim that every possible
vLLM control or workload scale has been exhausted.

The complete per-round optimization point and effect is in
[qwen38-b300-50-round-results.md](qwen38-b300-50-round-results.md). The
dashboard image is generated from the same ledger and the architecture image
shows the relationship between workload, serving profiles, evidence, quality
and the layered knowledge base.

## Harness lessons kept out of performance ranking

The first sweep attempt used the environment's pre-existing upstream `vllm`
console script. It bypassed the repository dispatcher, so the replay config was
parsed as an upstream vLLM benchmark type. I5–I9 remain as failed ledger
records; the corrected runner invokes
`agentinfer.agentcache.entrypoints.cli.main` through the active Python
environment and I10 is the executable smoke proof.

The GSM8K evaluator's parser is also part of the task contract: its endpoint
flag is `--endpoint` and its result-file flag is `--output`. Two rejected local
launch attempts used `--base-url`/`--output-dir`; they produced no result and
were not appended to the experiment ledger. I61 and I62 contain the corrected
commands and final quality evidence.

## Next controlled work

The next useful measurement is a clean vLLM-only environment, followed by a
larger trace sample and representative concurrency sweep. Kernel-level work is
eligible only after the serving profile is frozen: it must add a correctness
oracle, decode/prefill shape coverage, profiler or microbenchmark evidence and
an E2E replay confirmation. The vLLM skill rules, Z.ai dense-feedback model and
KDA task-contract discipline remain the procedure for those experiments; the
50-round numbers above are the model-specific evidence that determines
promotion.
