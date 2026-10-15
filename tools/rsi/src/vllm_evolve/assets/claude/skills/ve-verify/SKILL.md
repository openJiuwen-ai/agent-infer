---
name: ve-verify
description: Phase 4 (VERIFY) of a vllm-evolve round — run L1/L2 static safety + signature checks on the edited policy before spending a real benchmark on it.
---

# ve-verify — static safety gate

Run `ve phase set VERIFY`, then `ve verify <policy> <target>`.

This runs L1 (AST safety: no forbidden imports / patterns, return annotations)
and L2 (signature parity with the skeleton, nested-loop heuristic, fabricated-id
heuristic). If it reports issues, return to GENERATE, fix them, and re-verify.
Only a passing verify records `VERIFY_PASSED`, which `ve bench` requires — so you
cannot waste a real GPU run on a known-bad policy.

Next: the **ve-bench** skill.
