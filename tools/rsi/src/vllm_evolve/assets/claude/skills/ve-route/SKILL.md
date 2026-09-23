---
name: ve-route
description: Decompose a natural-language optimization request and route it to the right vllm-evolve mode (autopt / tune / port). Use this first when the user's intent is free-form and the mode is not yet decided.
---

# ve-route - intent routing (PROPOSAL only)

Turn a free-text request into a routing PROPOSAL, then hand off to the chosen mode.
This step never measures, judges, or runs a verb - it only classifies.

1. Read the request and classify the **mode**:
   - **autopt** - goal-directed automatic optimization (no specific policy given).
     The auto flow's internal *research* step (`ve-research` + `knowledge/`) runs
     inside autopt; `research` is NOT a top-level mode.
   - **tune** - the user named a specific policy `.py` to optimize/tune.
   - **port** - adapt / regression-check a policy across vLLM versions or hardware.
2. The deterministic router is `vllm_evolve.intent.router.route(text)`, returning
   `{mode, target, target_supported, policy, spec, ...}`. Trust it for the v1
   keyword/regex classification; only the `target=scheduling` surface is runnable.
3. Hand off:
   - autopt -> the `ve-autopt` skill (`ve autopt "<goal>"`).
   - tune -> the `ve-tune` skill (`ve tune <policy>`).
   - port -> the `ve-port` skill (`ve port <policy> --versions ... --hardware ...`).

If `target_supported` is false, stop and report it - only `scheduling` is wired to
the real backend.
