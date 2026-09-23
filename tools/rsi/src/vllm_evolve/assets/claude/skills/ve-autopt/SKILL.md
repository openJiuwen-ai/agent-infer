---
name: ve-autopt
description: Orchestrate the autopt optimization loop with Claude Code sub-agents — understand the goal, profile + diagnose on REAL vLLM, research the knowledge base, author a policy or pick a config knob, and adopt ONLY through the deterministic gate. Sub-agents propose; the gate decides.
---

# ve-autopt — sub-agent-driven optimization loop

You are the **orchestrator**. You supply NO judgment yourself: you run the deterministic `ve` verbs
as TOOLS, spawn specialized sub-agents (via the Task tool) for the intelligence, and let the gate
decide. A sub-agent's output is always a **PROPOSAL** — never a measurement, never a verdict.

## The loop (one round)

0. **Understand the goal** — spawn `ve-goal` on the user's intent → a `Spec` JSON (validate against
   `schemas/autopt/spec.schema.json`). The `accept_threshold_pct` is NOT taken from the agent — it
   comes from config / `--accept-threshold-pct`.

1. **Measure** — run `ve profile ...` (real vLLM; deterministic). This is the ONLY source of numbers.
   Off-box, use `--backend local_smoke` to dry-run the plumbing (its numbers can never adopt).

2. **Diagnose** — run `ve diagnose <profile.json>` (the deterministic rule-engine first opinion),
   then spawn `ve-diagnose` with the real profile + that first opinion → a richer `Diagnosis`
   (validate against `schemas/autopt/diagnosis.schema.json`). A wrong diagnosis is SAFE — it only
   steers the search; the gate still rejects any candidate without a proven gain.

3. **Research** — run `ve inspect --top <target>` to gather the knowledge base AND
   `ve experiment list` to gather the **hypothesis ledger** (the event-sourced research record).
   Spawn `ve-research` with BOTH; it synthesizes prior policy/lesson records and ledger entries,
   treating `supported`, `falsified`, and `inconclusive` hypotheses all as evidence — a `falsified`
   hypothesis is a known dead end to AVOID re-proposing (negative results are first-class). Before
   trusting its summary, VERIFY every citation: `ve inspect --verify-citations <ids>` — this checks
   both `policy_id`/`lesson_id` and `hypothesis:<event_id>` citations, and you reject the summary if
   `all_real` is false.

3b. **Free hypothesis → experiment (Research Harness / P4)** — if `ve-diagnose` attached a structured
   `{statement, prediction, experiment_spec}` proposal (a suspicion it wants MEASURED, not just
   prose), route it through the deterministic research chain: write the `experiment_spec` to a JSON
   file and run `ve experiment run <spec.json>`. That registers the prediction (prediction-first),
   runs the A/B experiment, and lets the **deterministic adjudicator** return
   `supported` / `falsified` / `inconclusive` — appended to the ledger as immutable events. YOU never
   decide the verdict; the adjudicator does, from the real measurement. The resulting
   `hypothesis:<event_id>` becomes citable research the NEXT round's `ve-research` can reuse. This is
   PROPOSAL-layer research evidence — it NEVER feeds the adoption gate (step 5 stays the only judge of
   a code/config win).

4. **Propose a change** — for a CONFIG target, pick a knob value (the deterministic `ve optimize`
   search). For a CODE target, spawn `ve-author` with the `AuthorContext.to_prompt()` rendering
   (diagnosis + parent scores + lessons + last verify errors); the agent RETURNS the child's
   source string — **YOU (the orchestrator) write it** into `targets/scheduling/work.py` in the
   GENERATE phase (the hook gates your pen; the author has none).

5. **Gate — deterministic; you do not judge here:**
   - `ve verify <policy> <target>` (static L1/L2) → must pass.
   - Bench the A/B at the SAME caliber but a DIFFERENT runner:
     `ve bench <candidate> --runner candidate` AND `ve bench <baseline> --runner strong_baseline`.
     Benching both with the same `--runner` compares a policy to itself (a fake ~0% gain) — never do
     that.
   - Bench BOTH legs AGAIN at a different operating point for the **holdout** (so a win must
     generalize), producing `<hb.json>` / `<hc.json>`.
   - `ve verify-gain <baseline.json> <candidate.json> --exempt-knob <searched>
     --holdout-baseline <hb.json> --holdout-candidate <hc.json>` → the VERDICT.
     A candidate adopts ONLY if verify-gain says `adopt` (effective + quality + gain + holdout +
     same-caliber + real source). You READ this verdict; you never produce it.

6. **Decide** — on `adopt`: `ve keep ...`; switch the config; if the bottleneck shifted, re-profile
   (back to step 1). Otherwise `ve discard --reason ...` and try the next target. When no target
   yields a proven gain, conclude **DoD-B** (no provable gain) — honestly, never a fabricated win.

## Frozen rules
- Sub-agents PROPOSE; the deterministic verbs (`ve profile` / `ve bench` / `ve verify-gain`) MEASURE
  and JUDGE. Never let a sub-agent's text stand in for a number or a verdict.
- An authored policy is UNTRUSTED until it passes the full real gate. Synthetic / `local_smoke` /
  marker-forging / unverified candidates can never adopt.
- Real vLLM only; never fabricate; box-gated steps stay box-gated.
