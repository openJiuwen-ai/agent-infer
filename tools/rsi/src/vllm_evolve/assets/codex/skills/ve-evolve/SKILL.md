---
name: ve-evolve
description: Run the local-only vllm-evolve generational search with real Frontier, official BurstGPT fragments, stress traces, held-out validation, and a sim-winner report.
---

# ve-evolve — Codex + local Frontier

Use this skill when evolving `schedule_batch` locally without a GPU.

1. Read `AGENTS.md`, the scheduling skeleton and the current `AuthorContext` contract.
   Run `ve-research` after context/diagnosis; freeze the resulting research snapshot.
2. Confirm `VE_FRONTIER_REPO` and `VE_FRONTIER_PYTHON`; the bridge patch must be applied.
3. Use the official BurstGPT CSV, never the tiny parser fixtures, and record its SHA256.
4. Require `ve init --client codex --check` to report `health=healthy`; existence-only status is
   insufficient. Run the exact local command (the default TTFT SLO is 200 ms):
   `ve frontier-evolve --burstgpt <official.csv> --out runs/frontier_local_evolution
   --goal "maximize goodput while preserving completion and TTFT SLO"
   --seeds 0,1,2 --fragment-size 32 --slo-ttft-ms 200 --max-num-seqs 4
   --generations 2 --population 5 --max-total-evals 10`.
5. The engine first materializes chronological train (first third), validation (middle third), and
   a regression window (first half of the final third). It adds separately labeled stress traces,
   evaluates FCFS/SJF/LJF/LIFO/current-seed baselines, then runs the generational author with parent
   scores, the frozen ExpertBrief, mechanism lineage, and verify feedback. Generation 0 seeds;
   later deterministic slots include both typed two-parent crossover and single-parent mutation.
6. Freeze the winner source and SHA before materializing the final test window (last sixth).
   Held-out uses paired seeds 0/1/2 and the strongest baseline independently for each scenario.
7. If held-out acceptance fails, preserve that run as negative evidence. Promote its failure
   regime into a future validation regression only, choose a previously unseen chronological
   window for the next final audit, design a structural mechanism, and rerun. Never reuse an
   already observed held-out window as a new final test. Do not lower the 3%/2-of-3/-2% gates or
   pick favorable seeds.
8. Deliver `request.json`, `context.json`, `diagnosis.json`, `state.json`, `events.jsonl`, research
   query/source/manifest/mechanism/ExpertBrief/snapshot artifacts, per-candidate manifests, exact
   `frontier_command.json` records, trace hashes, raw metric paths, failed-attempt history, and the
   report. `sim_winner/work_variant.py` exists only after every gate passes; otherwise require a
   truthful `no_winner/evidence.json` while preserving `selection/`. Record whether the author was
   `template` or `codex`; omission of `--author-command` is template, never Codex.

The author returns a complete source string and never writes files. The orchestrator writes and
verifies it. Do not spawn Claude sub-agents. Do not invoke SSH or a remote backend in this workflow.
Resume only with `--resume`; the frozen request fingerprint and every referenced artifact SHA must
match. A candidate failure is a recorded attempt, not a workflow terminal.

Frontier is an actual simulator run, not `local_smoke`, but the result remains search-only:
`source=frontier_sim`, `outcome_class=simulator_nonqualifying`. Never call it a production vLLM
gain or pass it to `ve keep`.

Originality is a human judgment. Do not add a novelty score or novelty gate; document the algorithm,
pseudocode, meaningful control-flow changes and ablation instead.

An acceptance pass requires all of: aggregate median goodput gain ≥3%, at least two of three
held-out scenarios positive, minimum scenario gain ≥-2%, no completion loss, valid markers with
non-fallback invocations, and a mechanism ablation. A simulator pass remains a proposal requiring
later real-vLLM verification.
