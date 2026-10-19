<!-- vllm-evolve-codex:start -->
## vllm-evolve local Frontier workflow

This project uses the canonical Codex integration as workflow controller and the real CPU Frontier
simulator as evaluator. The source author is recorded explicitly: the default is deterministic
`template`; `codex` requires an explicit author command. Never relabel template output as Codex.

### Execution persistence

Once the user gives an authorized goal and its acceptance criterion, keep advancing without asking
for confirmation between routine steps. Repeat the phase-appropriate
observe → diagnose → act → verify loop until the criterion is met or a true hard boundary is
reached. A failed candidate, invalid or underloaded benchmark, occupied resource, transient tool or
network error, exhausted search round, or negative result ends that attempt, not the user's goal:
preserve the evidence, choose the next safe in-scope action, and continue.

A hard boundary exists only when further progress requires new or materially expanded authority,
a credential or load-bearing input that cannot be discovered safely, a destructive/external action
outside the granted scope, or an externally controlled resource after all safe in-scope alternatives
have been exhausted. Never bypass a phase, protected path, frozen artifact, evidence rule, or
acceptance gate in order to keep moving. At a hard boundary, report the exact blocker and attempted
alternatives, then ask one minimal question that unlocks the already-prepared next action. Do not
pause to ask whether to continue, and treat non-blocking questions as optional while work proceeds.

- Use the `ve-evolve` skill for the complete local workflow.
- Default to `ve frontier-evolve`; do not invoke SSH, the remote backend, or real vLLM unless the
  user explicitly starts a separate real-hardware goal.
- Run the `ve-research` skill between diagnosis/context and design. Freeze one cited ExpertBrief
  per round; live failure must be reported as curated offline fallback.
- Treat `AuthorContext.to_prompt()` as the complete author contract: diagnosis, parent source and
  scores, peers, lessons, research_context, author_kind, verify errors, generation, and remaining
  budget must all be consumed.
- A Codex author returns source plus CandidateManifest; the orchestrator is the only writer and
  binds provenance hashes. Never label the deterministic template fallback as Codex.
- Frontier results are real simulator measurements but remain `frontier_sim`,
  `simulator_nonqualifying` sim-winner proposals.
- Evaluate against the strongest baseline on disjoint train/validation/held-out traces.
- Do not use a code-level novelty score. Explain the mechanism, code diff, and ablation for human
  review; measured simulator value controls the local result.
<!-- vllm-evolve-codex:end -->
