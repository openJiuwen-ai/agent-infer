---
name: ve-real-evolve
description: Run structural scheduling evolution with local Codex authoring and saturated remote real-vLLM evidence only.
---

# ve-real-evolve

Use a frozen `live-required` scheduling research snapshot whose mechanism cards prove a delta over
the pinned current vLLM main source. Calibrate and freeze a formal BurstGPT workload first.

Run:

```bash
ve phase set READ_CONTEXT
ve context scheduling
ve phase set DESIGN
ve design scheduling --note "<falsifiable live-vLLM mechanism>"
ve real-evolve \
  --bench-config <formal-strong-baseline.json> \
  --research-snapshot <research_snapshot.json> \
  --author-command "python -m vllm_evolve.engine.codex_author_cli" \
  --generations 2 --population 4 --max-total-evals 8 \
  --out <new-immutable-run-dir>
```

`ve real-evolve` is phase-locked to `DESIGN`. It internally performs per-child authoring, L1/L2
verification, and remote proposal evaluation, but it never adopts the selected child. A proposal
must then pass the formal three-scenario real-vLLM suite. Non-manual `ve keep` additionally requires
`--acceptance-evidence <suite-result.json>`; that evidence is policy-SHA-bound and must prove paired
seeds 0/1/2, valid saturated execution, exact completion, measured quality, and a supported
mechanism-control ablation. The suite invocation must declare every mechanism action with repeated
`--required-action-counter <plugin_provenance_key>` arguments; the raw evidence is re-evaluated at
keep time and every declared counter must fire in at least two scenarios. A lone
`source=real_vllm` eval result or precomputed `accepted=true` boolean is insufficient.

The local Codex author receives the complete AuthorContext and returns source plus
CandidateManifest. It must not write files or run remote tests. The orchestrator verifies and writes
each child, then evaluates it remotely. Never use Frontier or another simulator in this workflow.
Invalid load, incomplete requests, plugin fallback, missing mechanism actions, and underloaded GPU
all produce lineage with no fitness. Treat the winner as a proposal until paired held-out seeds and
the minimum source-level ablation pass the deterministic acceptance gate.
