---
name: ve-design
description: Phase 2 (DESIGN) of a vllm-evolve round — record one concrete, testable hypothesis for the policy change before writing any code.
---

# ve-design — record the hypothesis

Run `ve phase set DESIGN`, then `ve design --note "<hypothesis>"`.

Write **one** concrete, testable change and the reason you expect it to help a
specific metric (e.g. "prioritize short prompts to cut p99 TTFT under bursty
load"). Keep it to a single mechanism per round — small changes are easier to
attribute and verify. Do not edit the policy yet.

Next: the **ve-generate** skill.
