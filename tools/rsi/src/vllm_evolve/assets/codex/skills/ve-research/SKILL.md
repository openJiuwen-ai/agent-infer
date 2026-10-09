---
name: ve-research
description: Compile a cited target-aware expert brief from classic/recent primary sources, repository code, and immutable experiment memory before Codex designs an evolved inference mechanism.
---

# ve-research — compile temporary domain expertise

Use this skill after profiling/diagnosis and before `ve-design` or `ve-generate`.

1. Run `ve research build --target <target> --goal "<objective>" --diagnosis <diagnosis.json>
   --environment <environment.json> --out <run>/research`.
2. Use `--mode auto` for a live primary-source refresh with an explicit curated fallback, or
   `--mode offline` for a reproducible reviewed snapshot. Never describe an offline fallback as a
   current live search.
3. Read `expert_brief.json`, `mechanism_cards.json`, `source_manifest.json`, internal lessons, and
   falsified hypotheses. Verify the snapshot with
   `ve research verify <run>/research/research_snapshot.json`.
4. Apply compatibility filters: vLLM version, hardware, workload, available policy/plugin surface,
   and the cost of state movement/recomputation. A paper's claimed gain is not this run's evidence.
5. Select a portfolio of structurally different mechanisms. For every candidate preserve cited
   mechanism IDs, a falsifiable hypothesis, affected code symbols, risks, and required ablations in
   `CandidateManifest`.
6. Freeze the snapshot for the round. Refresh only after the bottleneck changes, evidence falsifies
   a key assumption, or the user explicitly asks.

The research layer is an untrusted proposal producer. Frontier ranks simulator candidates;
real-vLLM measurement and the existing deterministic gate alone can establish/adopt a gain.
