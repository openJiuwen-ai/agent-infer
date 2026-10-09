---
name: ve-generate
description: Return complete schedule_batch source from the full AuthorContext prompt.
---

# ve-generate — author source

Consume the complete `AuthorContext.to_prompt()` JSON: doctrine, spec, diagnosis, skeleton,
parent source and score, peers, lessons, `research_context`, `author_kind`, last errors,
`operator`, generation, and remaining budget. For crossover, `parents` contains exactly the two
selected distinct verified/scored parents, including complete sources, scores, manifests, evidence,
and diagnostics. Consume both.

For a Codex author return one JSON object with:

- `source`: complete Python source defining the required `schedule_batch` signature;
- `manifest`: CandidateManifest with cited mechanism IDs, hypothesis, structural change, affected
  symbols, expected regimes, risks, ablations, parameter-only flag, and snapshot hash.

Do not write candidate files from the author: the evolution orchestrator is the only writer and
binds source/parent/snapshot hashes. For a manual
phase-locked edit, first run `ve phase set GENERATE` and edit only
`targets/scheduling/work.py` inside the evolution block. Continue with `ve-verify`.

For mutation, transform the one selected parent. For crossover, compose typed scheduling semantics
and return one complete child; do not splice source text, copy a parent score, ignore either parent,
or return either parent unchanged. The orchestrator binds ordered `parent_shas`, inherited
mechanisms, and provenance. The manifest must also name concrete `inherited_components` from each
parent, compatibility, combined control flow, and parent-specific ablation plans. Verification
or author failures enter bounded repair and do not end the user's goal.

Do not label a deterministic template proxy as an implemented paper mechanism. Research influence
is recorded separately as `research_inspirations`; only source-level behavior actually implemented
by the child belongs in `mechanism_ids`.
