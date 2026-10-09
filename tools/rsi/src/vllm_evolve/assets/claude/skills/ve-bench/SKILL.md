---
name: ve-bench
description: Phase 5 (BENCHMARK) of a vllm-evolve round — run the candidate on REAL vLLM (Docker) under a load profile and capture metrics. Requires a GPU host.
---

# ve-bench — real-vLLM benchmark

Run `ve phase set BENCHMARK` (only reachable after `VERIFY_PASSED`), then
`ve bench <policy> --profile <profile>` for the profile(s) you care about:

- **throughput** — saturating load; primary metric is SLO-meeting goodput.
- **latency** — low load, tight SLO; single-request responsiveness.
- **replay** — a real trace (BurstGPT / Azure / ShareGPT / Mooncake).

This launches a pinned vLLM container, drives the load, and records
`eval_result.json` (per-seed metrics + median + CV). It needs a GPU + Docker; it
will refuse honestly off-hardware rather than fake numbers. Run the seed too, to
get a baseline eval_result for comparison.

Next: the **ve-decide** skill.
