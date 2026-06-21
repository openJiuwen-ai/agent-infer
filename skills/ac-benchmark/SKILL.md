---
name: ac-benchmark
description: Use when measuring AgentCache cache hit rate, latency, or throughput against a target engine (vLLM or in-process stub); bootstraps the harness layout if absent and compares against the baseline.
---

# ac-benchmark

Run the AgentCache benchmark harness against a target LLM serving engine,
capture metrics, and compare against the recorded baseline.

## WHEN TO INVOKE

- The user asks to "benchmark AgentCache", "measure cache hit rate", "compare
  latency/throughput vs vLLM", or "run the benchmark suite".
- A change to cache behaviour needs evidence before merge.

Do NOT invoke for: unit-testing a function, profiling unrelated code, or
comparing two non-baseline runs against each other.

## STEPS

1. Locate the benchmark harness under `benchmarks/`.

   - **Bootstrap branch (harness does not exist yet):** create the layout
     first, then continue to step 2:

     ```bash
     mkdir -p benchmarks/results
     touch benchmarks/results/.gitkeep
     cat > benchmarks/baseline.json <<'JSON'
     {
       "hit_rate": null,
       "p50_latency_ms": null,
       "p99_latency_ms": null,
       "throughput_rps": null,
       "note": "populate with the first measured run"
     }
     JSON
     printf '"""AgentCache benchmark harness entry point."""\n' \
       > benchmarks/run.py
     ```

2. Confirm a target is reachable:

   - vLLM: a base URL is set (`export AGENTCACHE_VLLM_URL=...`) and reachable.
   - Stub: an in-process stub target is configured (no external dependency).

3. Run a warmup pass to populate the cache:

   ```bash
   python benchmarks/run.py --target "$AGENTCACHE_VLLM_URL" --warmup
   ```

4. Run the measured pass, writing JSON metrics to a timestamped result file:

   ```bash
   python benchmarks/run.py --target "$AGENTCACHE_VLLM_URL" \
     --out "benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ).json"
   ```

5. Compare the new result against `benchmarks/baseline.json` (hit-rate,
   p50/p99 latency, throughput). Report each metric's delta and direction.
6. If the run regresses any metric relative to baseline, state so explicitly
   (do not silently commit a worse number as the new baseline).

## DON'T

- Don't compare against a non-baseline run as if it were the baseline.
- Don't delete old `benchmarks/results/*.json` files — they are the history.
- Don't trust a single iteration; the harness must take multiple samples and
  report percentiles.

## AFTER

Once metrics are captured and compared, invoke `ac-review` to validate any
harness or baseline changes before opening the PR.
