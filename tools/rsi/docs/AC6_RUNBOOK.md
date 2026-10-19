# AC4 → AC5 → AC6 Real-Validation Runbook (BOX-GATED)

This runbook is the exact, ordered sequence to produce the **real** result on the GPU box
(`gpu-host`). Real AC7/AC8 evidence is box-gated and cannot be produced off-hardware.

> **`local_smoke` is NOT this.** The laptop path (`ar bench`/`ar autopt --backend local_smoke`) runs
> the whole flow off-box with SYNTHETIC numbers for plumbing only. A `local_smoke` result can NEVER
> be a real gain / keep / AC6 result — it is quarantined by four locks (source / outcome_class /
> effective+quality / the LOCK-D real-source guard at compare/verify-gain/keep/accept) and can only
> ever conclude DoD-B. Everything below requires `--backend remote` (the default) on the real box.

## Iron rules (never relax)
- **Real vLLM only. Never fabricate.** Every number comes from `ar bench` against the real server.
  A profiled CUDA window is *diagnostic only* — never used as a performance number.
- **The opponent is the STRONG baseline** (vanilla + known optimizations, auto-tuned + frozen),
  not naive vanilla. The contribution is a **new algorithm** on the diagnosed bottleneck surface.
- **Honest DoD-B** is a valid, expected outcome: if the bottleneck is standard config optimization
  (e.g. fp8) rather than a code surface, say so — do not invent a scheduling win.
- Adoption requires ALL of: effective (real probe) + measured quality non-regression + paired
  bootstrap `CI_low ≥ X%` + frozen holdout + `same_caliber`. Anything missing ⇒ no adoption.

## 0. Preflight (must pass first)
```
python -m vllm_evolve.tools.box_preflight --remote gpu-host
```
If `box_unreachable`: check your local SSH configuration and host availability, then retry.
Never put passwords or private keys in reports or the repository. **Do not proceed.**

## 1. AC2/AC3 — calibrate + freeze the strong baseline
```
ar phase set BENCHMARK
ar calibrate --model <7B> --gpus 0 --remote gpu-host --out runs/strong_baseline.json
```
Drives the rising closed-loop sweep, locks the saturated single-card max-throughput operating
point, and writes the **frozen** `strong_baseline.json` (locked BenchConfig + eval_result + per-seed
tok/s + rendered/remote cmd). This frozen artifact is the binding A/B opponent for everything below.

## 2. AC4 — real CUDA diagnosis (same-caliber)
```
ar profile --model <7B> --remote gpu-host --cuda nsys --diagnose --out runs/profile.json
```
Runs the same-caliber `python -m vllm_evolve.bench.native` workload under nsys, pulls the artifact
back, parses it into a `KernelBreakdown`, and feeds it to KernelToLeverMap → the bottleneck surface
+ allowed levers. (Unprofiled metrics come from step 1, not the profiled window.)

## 3. AC5 — candidate on the diagnosed surface (or honest DoD-B)
- **If KernelToLeverMap points to a scheduling code surface** (free SM capacity + standing queue):
  evolve/implement ONE candidate in `targets/scheduling/work.py`, then:
  ```
  ar phase set GENERATE   # edit targets/scheduling/work.py only here
  ar phase set VERIFY     && ar verify work.py scheduling
  ar phase set BENCHMARK  && ar bench work.py scheduling --runner candidate --remote gpu-host --out runs/cand.json
  ```
  The candidate must pass the **effective** gate (out-of-process probe: invoked + changed order +
  no fallback) and the **measured fixed-set quality** gate. A candidate that falls back or regresses
  quality is rejected.
- **If the bottleneck is config-only** (compute/memory → fp8 / cuda-graph, not code): **stop and
  record an honest DoD-B** — the gain is a known config lever, not a new algorithm.

## 4. AC6 — strict validation vs the frozen strong baseline
```
ar phase set KEEP_OR_DISCARD
ar compare runs/strong_baseline.json runs/cand.json        # same_caliber-gated
ar verify-gain runs/strong_baseline.json runs/cand.json \
    --metric output_throughput_tok_s --accept-threshold-pct <X> \
    --holdout-baseline runs/sb_holdout.json --holdout-candidate runs/cand_holdout.json
```
`ar verify-gain` enforces: `same_caliber` (else `same_caliber_unverifiable`/`mismatch`), measured
`quality_ok`, paired bootstrap on per-seed deltas (`point ≥ X%` AND `CI_low ≥ X%`), and the frozen
holdout. `adopt: true` only when all hold.

## 5. Emit the verdict
- **Adopted** → report the measured **+X% over the strong baseline** with CI, holdout, quality,
  and effective evidence.
- **Not adopted** → honest **DoD-B**: how far it got, why, the bottleneck, and the ceiling evidence.

Either outcome is a real, non-fabricated result — which is the whole point.
