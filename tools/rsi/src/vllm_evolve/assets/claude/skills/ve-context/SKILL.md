---
name: ve-context
description: Phase 1 (READ_CONTEXT) of a vllm-evolve round — gather the target's skeleton, seed, prompt hints, and best-so-far before designing a change.
---

# ve-context — gather evolution context

Run `ve phase set READ_CONTEXT`, then `ve context <target>` (default
`scheduling`). From the JSON, study:

- **skeleton** — the function contract you must preserve (signature, types).
- **seed** — the current baseline policy you are trying to beat.
- **prompt_hints** — domain knowledge about this target.
- **best** — top policies recorded so far (don't repeat their ideas blindly).

Identify the single biggest opportunity. Do **not** edit anything in this phase.
Next: the **ve-design** skill.
