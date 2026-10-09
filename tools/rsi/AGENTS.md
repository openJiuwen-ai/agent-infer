# vllm-evolve — Agent Guide

The repository supports Codex and Claude Code. The shared implementation contract is the `ve` CLI,
the phase/protected-path guard, `AuthorContext.to_prompt()`, and the frozen verification/adoption
boundary. Client assets must not fork those semantics.

## Execution persistence

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

## Default Codex workflow: local Frontier evolution

Use the packaged `ve-evolve` skill for a local CPU search:

```bash
ve init --client codex
ve frontier-evolve \
  --burstgpt /absolute/path/to/BurstGPT_1.csv \
  --out runs/frontier_local_evolution \
  --seeds 0,1,2
```

This workflow must:

- use `VE_FRONTIER_REPO` and `VE_FRONTIER_PYTHON` to execute the real local
  `python -m frontier.main` subprocess;
- never invoke SSH, a remote backend, GPU, or real vLLM unless the user starts a separate
  real-hardware task;
- consume every field rendered by `AuthorContext.to_prompt()`: diagnosis, parent source/scores,
  peers, lessons, frozen research context, author kind, last errors, generation, and budget;
- run target-aware `ve research build` after diagnosis/context and before design; keep classic and
  recent primary sources, repository implementation hooks, internal lessons, and falsified
  hypotheses in separate evidence fields;
- freeze one research snapshot per round and report an online failure as an offline curated
  fallback, never as a live/latest search;
- require a Codex author to return complete policy source plus `CandidateManifest`; only the
  orchestrator binds hashes and writes candidate files, and a deterministic template fallback must
  remain labeled `author_kind=template`;
- run L1/L2 verification before Frontier;
- require marker SHA equality, `invocations > 0`, and `fallbacks < invocations`;
- keep official BurstGPT fragments separate from synthetic multi-tenant/prefix stress data;
- hide the chronological test fragment until the winner source is frozen;
- compare against the strongest per-scenario FCFS/SJF/LJF/LIFO/current-seed result with paired
  seeds;
- preserve the 3% median / 2-of-3 positive / -2% floor / completion / marker / ablation gates.

Do not add an automatic novelty score or gate. Human review of the mechanism, meaningful
control-flow change, pseudocode, source diff, failure history, and ablation determines whether an
algorithm is genuinely interesting.

Frontier evidence is real simulator output but always remains `source=frontier_sim`,
`outcome_class=simulator_nonqualifying`. It can create only `sim_winner`; it cannot be passed to
`ve keep`, called a production gain, or used as real-vLLM acceptance evidence.

## Production real-vLLM workflow

For a separately authorized GPU/remote task, use the phase-locked round:

```text
ve context → ve design → edit policy → ve verify → ve bench
  → ve compare → ve keep / ve discard
```

Only clean `source=real_vllm` evidence may cross the production adoption boundary. `local_smoke` is
synthetic plumbing only. Do not fabricate metrics or reinterpret either nonqualifying source.

## Protected state

The phase hook blocks edits to frozen seeds, skeletons, benchmark configs, schemas, the archive,
`.ve` state, and run provenance. Change packaged Codex assets under
`src/vllm_evolve/assets/codex/`, not the generated `.codex/` or `.agents/` copy. The Claude guide
remains in [CLAUDE.md](CLAUDE.md); it does not override this shared evidence doctrine.

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
