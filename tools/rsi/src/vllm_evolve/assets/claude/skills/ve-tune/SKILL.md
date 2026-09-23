---
name: ve-tune
description: Tune a GIVEN scheduling policy's serving config on real vLLM (mode b). Use when the user supplies a specific policy .py and wants it optimized/tuned rather than evolved from scratch.
---

# ve-tune - tune a given policy

Optimize the serving config around a supplied policy, reusing the same gated
pipeline as autopt (profile -> config search -> bootstrap+holdout verify). It does
NOT evolve new code - it tunes the policy you give it.

1. Run `ve tune <policy.py> [scheduling] --goal "<objective>"`.
   - Add `--backend local_smoke` to exercise the plumbing off-box (synthetic;
     can only conclude DoD-B, never a kept gain).
   - Real tuning is the default `remote` backend (real vLLM, GPU box).
2. Adoption stays in the frozen accept gate: a candidate is kept only via a real
   `verify-gain` / `keep` with a clean `real_vllm` eval_result. A `local_smoke`
   result can never become a gain/keep.
3. Only `scheduling` is supported; any other target is refused.

Never claim an improvement without a real bootstrap+holdout verdict.
