---
name: ve-port
description: Adapt / regression-check a scheduling policy across vLLM versions and hardware (mode c). Use when an algorithm must be debugged, adapted, or perf-checked between different vLLM versions or GPUs.
---

# ve-port - version x hardware matrix

Evaluate one policy across a matrix of vLLM versions and hardware, producing a
`port_report.json`. This is a **report**, not an adoption path - it never keeps or
adopts; promotion of any winning cell still goes through the normal keep gate.

1. Run `ve port <policy.py> [scheduling] --versions <v1,v2> --hardware <h1,h2>`.
   - `--backend local_smoke` validates the matrix plumbing off-box (each cell is
     synthetic `source=local_smoke`, non-promotable).
   - Real per-cell benches run on the GPU box (`@requires_gpu`); off-box they skip.
2. Read `port_report.json`: per-cell `{vllm_version, hardware_id, status, source,
   outcome_class, regression_verdict}` plus a summary. Use it to spot which
   version/hardware cells regress or need adaptation.
3. Only `scheduling` is supported; a missing policy file or other target is refused.

Deep per-cell regression diffing and adaptation patches are box-gated follow-ups.
