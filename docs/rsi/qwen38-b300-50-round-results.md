# Qwen3.8-27B B300 RSI: round-level result extract

This generated extract is derived from the append-only ledger at
`/home/zjy/code/hsliu/tmp/rsi-pr25-b300-20260922/rsi-real/experiments.jsonl`.
The corrected 50-round sweep is I11–I60. I5–I9 are preserved harness failures
from the pre-fix console entry point; I63–I67 are a post-sweep combination check.
The raw result directories and evidence hashes remain in the ledger.

## Frozen contract

Seed 228, two replay tasks, concurrency 2, TP4 on GPUs 0–3, BF16 weights,
prefix caching, max context 262144, exact prompt calibration tolerance 0,
`inferact_codex_swebenchpro`, and the same cached conversion bundle.

## Profile effect

| Profile | Valid | Median tok/s | Range tok/s | Delta vs baseline | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `async-off` | 5/5 | 186.674 | 182.972–188.543 | -10.40% | measured; no promotion |
| `async-on` | 5/5 | 209.065 | 206.643–211.929 | +0.34% | measured; no promotion |
| `baseline-current` | 5/5 | 208.347 | 205.072–212.562 | +0.00% | measured; no promotion |
| `batch-32768` | 5/5 | 210.385 | 202.529–213.529 | +0.98% | quality-qualified candidate |
| `batch-65536` | 5/5 | 208.680 | 201.952–212.749 | +0.16% | measured; no promotion |
| `batch-8192` | 1/5 | 210.503 | 210.503–210.503 | +1.03% | coverage rejected |
| `batch32768-async-on` | 5/5 | 208.134 | 201.165–210.278 | -0.10% | measured; no promotion |
| `kv-fp8` | 5/5 | 210.658 | 203.665–213.419 | +1.11% | quality rejected (I61) |
| `max-seqs-8` | 5/5 | 206.784 | 204.168–211.157 | -0.75% | measured; no promotion |
| `no-prefix-cache` | 5/5 | 143.964 | 143.515–146.455 | -30.90% | measured; no promotion |
| `stream-8` | 5/5 | 208.524 | 203.478–212.204 | +0.08% | measured; no promotion |

## Every replay iteration

| Round | Profile | Optimization point | Status | Effect |
| --- | --- | --- | --- | --- |
| I11 | `baseline-current` | Repeat the I2 server arguments without changing a serving knob. | `measured` | 208.309 tok/s (-0.02% vs baseline median); 36/36, residual 0 |
| I12 | `baseline-current` | Repeat the I2 server arguments without changing a serving knob. | `measured` | 208.347 tok/s (+0.00% vs baseline median); 36/36, residual 0 |
| I13 | `baseline-current` | Repeat the I2 server arguments without changing a serving knob. | `measured` | 210.392 tok/s (+0.98% vs baseline median); 36/36, residual 0 |
| I14 | `baseline-current` | Repeat the I2 server arguments without changing a serving knob. | `measured` | 205.072 tok/s (-1.57% vs baseline median); 36/36, residual 0 |
| I15 | `baseline-current` | Repeat the I2 server arguments without changing a serving knob. | `measured` | 212.562 tok/s (+2.02% vs baseline median); 36/36, residual 0 |
| I16 | `no-prefix-cache` | Disable prefix caching while keeping the model, TP4, scheduler and cache budget fixed. | `measured` | 146.455 tok/s (-29.71% vs baseline median); 36/36, residual 0 |
| I17 | `no-prefix-cache` | Disable prefix caching while keeping the model, TP4, scheduler and cache budget fixed. | `measured` | 145.733 tok/s (-30.05% vs baseline median); 36/36, residual 0 |
| I18 | `no-prefix-cache` | Disable prefix caching while keeping the model, TP4, scheduler and cache budget fixed. | `measured` | 143.964 tok/s (-30.90% vs baseline median); 36/36, residual 0 |
| I19 | `no-prefix-cache` | Disable prefix caching while keeping the model, TP4, scheduler and cache budget fixed. | `measured` | 143.541 tok/s (-31.10% vs baseline median); 36/36, residual 0 |
| I20 | `no-prefix-cache` | Disable prefix caching while keeping the model, TP4, scheduler and cache budget fixed. | `measured` | 143.515 tok/s (-31.12% vs baseline median); 36/36, residual 0 |
| I21 | `batch-8192` | Set --max-num-batched-tokens=8192; keep chunked prefill enabled by vLLM defaults. | `measured` | 210.503 tok/s (+1.03% vs baseline median); 36/36, residual 0 |
| I22 | `batch-8192` | Set --max-num-batched-tokens=8192; keep chunked prefill enabled by vLLM defaults. | `failed` | coverage 27/36; rejected (Incomplete replay coverage: planned=36, successful=27, failed=0) |
| I23 | `batch-8192` | Set --max-num-batched-tokens=8192; keep chunked prefill enabled by vLLM defaults. | `failed` | coverage 27/36; rejected (Incomplete replay coverage: planned=36, successful=27, failed=0) |
| I24 | `batch-8192` | Set --max-num-batched-tokens=8192; keep chunked prefill enabled by vLLM defaults. | `failed` | coverage 27/36; rejected (Incomplete replay coverage: planned=36, successful=27, failed=0) |
| I25 | `batch-8192` | Set --max-num-batched-tokens=8192; keep chunked prefill enabled by vLLM defaults. | `failed` | coverage 27/36; rejected (Incomplete replay coverage: planned=36, successful=27, failed=0) |
| I26 | `batch-32768` | Set --max-num-batched-tokens=32768. | `measured` | 206.302 tok/s (-0.98% vs baseline median); 36/36, residual 0 |
| I27 | `batch-32768` | Set --max-num-batched-tokens=32768. | `measured` | 210.385 tok/s (+0.98% vs baseline median); 36/36, residual 0 |
| I28 | `batch-32768` | Set --max-num-batched-tokens=32768. | `measured` | 202.529 tok/s (-2.79% vs baseline median); 36/36, residual 0 |
| I29 | `batch-32768` | Set --max-num-batched-tokens=32768. | `measured` | 210.521 tok/s (+1.04% vs baseline median); 36/36, residual 0 |
| I30 | `batch-32768` | Set --max-num-batched-tokens=32768. | `measured` | 213.529 tok/s (+2.49% vs baseline median); 36/36, residual 0 |
| I31 | `batch-65536` | Set --max-num-batched-tokens=65536. | `measured` | 208.706 tok/s (+0.17% vs baseline median); 36/36, residual 0 |
| I32 | `batch-65536` | Set --max-num-batched-tokens=65536. | `measured` | 207.917 tok/s (-0.21% vs baseline median); 36/36, residual 0 |
| I33 | `batch-65536` | Set --max-num-batched-tokens=65536. | `measured` | 208.680 tok/s (+0.16% vs baseline median); 36/36, residual 0 |
| I34 | `batch-65536` | Set --max-num-batched-tokens=65536. | `measured` | 201.952 tok/s (-3.07% vs baseline median); 36/36, residual 0 |
| I35 | `batch-65536` | Set --max-num-batched-tokens=65536. | `measured` | 212.749 tok/s (+2.11% vs baseline median); 36/36, residual 0 |
| I36 | `async-on` | Enable --async-scheduling explicitly. | `measured` | 206.643 tok/s (-0.82% vs baseline median); 36/36, residual 0 |
| I37 | `async-on` | Enable --async-scheduling explicitly. | `measured` | 207.644 tok/s (-0.34% vs baseline median); 36/36, residual 0 |
| I38 | `async-on` | Enable --async-scheduling explicitly. | `measured` | 209.106 tok/s (+0.36% vs baseline median); 36/36, residual 0 |
| I39 | `async-on` | Enable --async-scheduling explicitly. | `measured` | 211.929 tok/s (+1.72% vs baseline median); 36/36, residual 0 |
| I40 | `async-on` | Enable --async-scheduling explicitly. | `measured` | 209.065 tok/s (+0.34% vs baseline median); 36/36, residual 0 |
| I41 | `async-off` | Disable --async-scheduling explicitly. | `measured` | 188.160 tok/s (-9.69% vs baseline median); 36/36, residual 0 |
| I42 | `async-off` | Disable --async-scheduling explicitly. | `measured` | 183.406 tok/s (-11.97% vs baseline median); 36/36, residual 0 |
| I43 | `async-off` | Disable --async-scheduling explicitly. | `measured` | 182.972 tok/s (-12.18% vs baseline median); 36/36, residual 0 |
| I44 | `async-off` | Disable --async-scheduling explicitly. | `measured` | 188.543 tok/s (-9.51% vs baseline median); 36/36, residual 0 |
| I45 | `async-off` | Disable --async-scheduling explicitly. | `measured` | 186.674 tok/s (-10.40% vs baseline median); 36/36, residual 0 |
| I46 | `stream-8` | Set --stream-interval=8. | `measured` | 209.305 tok/s (+0.46% vs baseline median); 36/36, residual 0 |
| I47 | `stream-8` | Set --stream-interval=8. | `measured` | 208.524 tok/s (+0.08% vs baseline median); 36/36, residual 0 |
| I48 | `stream-8` | Set --stream-interval=8. | `measured` | 206.573 tok/s (-0.85% vs baseline median); 36/36, residual 0 |
| I49 | `stream-8` | Set --stream-interval=8. | `measured` | 212.204 tok/s (+1.85% vs baseline median); 36/36, residual 0 |
| I50 | `stream-8` | Set --stream-interval=8. | `measured` | 203.478 tok/s (-2.34% vs baseline median); 36/36, residual 0 |
| I51 | `max-seqs-8` | Set --max-num-seqs=8 while the replay client remains at concurrency two. | `measured` | 206.517 tok/s (-0.88% vs baseline median); 36/36, residual 0 |
| I52 | `max-seqs-8` | Set --max-num-seqs=8 while the replay client remains at concurrency two. | `measured` | 211.157 tok/s (+1.35% vs baseline median); 36/36, residual 0 |
| I53 | `max-seqs-8` | Set --max-num-seqs=8 while the replay client remains at concurrency two. | `measured` | 206.784 tok/s (-0.75% vs baseline median); 36/36, residual 0 |
| I54 | `max-seqs-8` | Set --max-num-seqs=8 while the replay client remains at concurrency two. | `measured` | 204.168 tok/s (-2.01% vs baseline median); 36/36, residual 0 |
| I55 | `max-seqs-8` | Set --max-num-seqs=8 while the replay client remains at concurrency two. | `measured` | 207.195 tok/s (-0.55% vs baseline median); 36/36, residual 0 |
| I56 | `kv-fp8` | Set --kv-cache-dtype=fp8_e4m3; the candidate must pass GSM8K before promotion. | `measured` | 203.665 tok/s (-2.25% vs baseline median); 36/36, residual 0 |
| I57 | `kv-fp8` | Set --kv-cache-dtype=fp8_e4m3; the candidate must pass GSM8K before promotion. | `measured` | 206.257 tok/s (-1.00% vs baseline median); 36/36, residual 0 |
| I58 | `kv-fp8` | Set --kv-cache-dtype=fp8_e4m3; the candidate must pass GSM8K before promotion. | `measured` | 210.658 tok/s (+1.11% vs baseline median); 36/36, residual 0 |
| I59 | `kv-fp8` | Set --kv-cache-dtype=fp8_e4m3; the candidate must pass GSM8K before promotion. | `measured` | 213.419 tok/s (+2.43% vs baseline median); 36/36, residual 0 |
| I60 | `kv-fp8` | Set --kv-cache-dtype=fp8_e4m3; the candidate must pass GSM8K before promotion. | `measured` | 213.319 tok/s (+2.39% vs baseline median); 36/36, residual 0 |
| I63 | `batch32768-async-on` | Set --max-num-batched-tokens=32768 and --async-scheduling together; retain TP4, prefix caching and all frozen replay controls. | `measured` | 208.134 tok/s (-0.10% vs baseline median); 36/36, residual 0 |
| I64 | `batch32768-async-on` | Set --max-num-batched-tokens=32768 and --async-scheduling together; retain TP4, prefix caching and all frozen replay controls. | `measured` | 201.165 tok/s (-3.45% vs baseline median); 36/36, residual 0 |
| I65 | `batch32768-async-on` | Set --max-num-batched-tokens=32768 and --async-scheduling together; retain TP4, prefix caching and all frozen replay controls. | `measured` | 210.278 tok/s (+0.93% vs baseline median); 36/36, residual 0 |
| I66 | `batch32768-async-on` | Set --max-num-batched-tokens=32768 and --async-scheduling together; retain TP4, prefix caching and all frozen replay controls. | `measured` | 204.441 tok/s (-1.87% vs baseline median); 36/36, residual 0 |
| I67 | `batch32768-async-on` | Set --max-num-batched-tokens=32768 and --async-scheduling together; retain TP4, prefix caching and all frozen replay controls. | `measured` | 210.005 tok/s (+0.80% vs baseline median); 36/36, residual 0 |

## Quality gates

| Round | Candidate | Correct | Accuracy | Decision |
| --- | --- | ---: | ---: | --- |
| I4 | BF16 TP4 baseline | 1254/1319 | 95.072% | reference |
| I61 | FP8 KV (`fp8_e4m3`) | 1251/1319 | 94.845% | reject; below BF16 reference |
| I62 | BF16 + `max-num-batched-tokens=32768` | 1255/1319 | 95.148% | pass |

The measured winner under the current contract is BF16 TP4 with prefix caching,
`--max-num-batched-tokens=32768`, and 210.385 median output tok/s. The
`batch32768 + async-on` combination check (I63–I67) had a 208.134 median,
so its controls are not combined for the current promotion.
