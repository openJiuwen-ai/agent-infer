---
name: ve-context
description: Read the scheduling contract, seed, evidence, and lessons before a Codex-authored evolution.
---

# ve-context — gather evidence

For a manual phase-locked round, run `ve phase set READ_CONTEXT`, then
`ve context scheduling`. Treat `runtime_contract.contract_sha256` as the scheduling interface
identity. Read both `static_skeleton` and the rendered Frontier runtime decision/request contract;
the latter exposes `defer_ids` and `mechanism_applicable` and labels placeholder signals. Read the
current seed, prompt hints, prior winners, cited research, and lessons. Do not edit policy source in
this phase. Continue with `ve-research`;
static `prompt_hints.md` is orientation, not a substitute for a frozen ExpertBrief.

For `ve frontier-evolve`, the orchestrator supplies these facts through
`AuthorContext.to_prompt()` and preserves the simulator-only doctrine. Confirm the prompt includes
`research_context`, `author_kind`, `operator`, source citations, mechanism cards, implementation
hooks, a non-empty runtime contract, and the research snapshot hash. Continue with `ve-research`,
then `ve-design`.
