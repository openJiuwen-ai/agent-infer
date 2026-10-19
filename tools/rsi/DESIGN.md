# vllm-evolve — System Design

A Codex- and Claude-compatible harness for discovering better **vLLM serving policies** by evolving
policy code and proving it better on **real vLLM**. This document describes the *current*
architecture. For day-to-day usage see `CLAUDE.md` / `AGENTS.md` / `README.md`.

> Local Frontier is a real out-of-process CPU simulator used for candidate search, ablation, and
> falsification, but its evidence is permanently `simulator_nonqualifying`. The production adoption
> boundary accepts only clean real-vLLM measurements. `local_smoke` is plumbing-only. The one
> supported optimization target is `scheduling`.

## 1. The `ve` CLI + the phase machine

All work flows through the `ve` CLI (`src/vllm_evolve/cli/main.py`) under a **phase machine**
(`src/vllm_evolve/tools/phase_guard.py`) enforced by a PreToolUse hook. One round walks an ordered,
guarded chain of phases:

```
READ_CONTEXT -> DESIGN -> GENERATE -> VERIFY -> (VERIFY_PASSED) -> BENCHMARK
             -> KEEP_OR_DISCARD -> COMMIT_OR_ROLLBACK
```

Each verb is registered with the phase it is legal in (`_VERB_PHASES`); both the CLI and the hook
read that single source of truth, so a verb can only run in its phase. Policy files may change ONLY
in `GENERATE` (the hook blocks edits to seeds / skeletons / `config/` / `schemas/` /
`archive_policies/` so an evolution round cannot cheat). Verbs:

| Verb | Phase | What |
|------|-------|------|
| `ve phase set/status` | any | drive / inspect the phase machine |
| `ve context` | READ_CONTEXT | emit the `scheduling` skeleton + seed + hints + best-so-far |
| `ve design` | DESIGN | record one testable hypothesis |
| `ve verify` | VERIFY | L1 AST safety + L2 signature check → VERIFY_PASSED |
| `ve bench` | VERIFY_PASSED | run the real (or `local_smoke`) bench for a runner kind |
| `ve compare` | KEEP_OR_DISCARD | same-caliber A/B verdict (better / worse / inconclusive) |
| `ve keep` / `ve discard` | COMMIT_OR_ROLLBACK | archive a kept winner / audit a discard |

The `autopt` layer adds `ve goal / profile / diagnose / targets / optimize / calibrate /
verify-gain / autopt` for the L1→L4 optimization loop. `ve init` installs the agent + skills + hook
into a project's `.claude/`.

Only `scheduling` is wired to the real backend: `ve context` / `verify` / `bench` reject any other
target with `unsupported_target`.

## 2. The execution contract: `BenchConfig`

`src/vllm_evolve/bench/config.py` defines `BenchConfig` — the single source of truth for one
execution (engine levers, workload, runner kind, statistical config, environment). It renders the
serve command (`to_serve_args`), serves the artifact (`model_to_serve`), and carries `provenance()`
into every eval_result. `same_caliber(a, b)` proves an A/B differs only by the thing under test.

## 3. Backends

A single seam (`--backend {remote,local_smoke}` / `VE_BENCH_BACKEND`, default `remote`) selects:

- **`remote`** — the real path: `bench/dispatch.py` scp/ssh's the policy to the GPU box and runs
  `python -m vllm_evolve.bench.native`, which serves vLLM once and drives each seed's load, then
  serializes a schema-valid `eval_result` (`source='real_vllm'`). When the box is unreachable it
  reports `box_gated_blocked` with provenance — never a fabricated pass.
- **`local_smoke`** — `bench/local_smoke.py`, an in-process synthetic backend so the WHOLE flow runs
  on a laptop with no GPU. It is quarantined by FOUR independent locks: `source='local_smoke'`; a
  failure-class `outcome_class`; `effective=False` / `quality_ok=None` and it never emits the plugin
  marker; and **LOCK D** — a shared real-source guard (`bench/eval_result.real_source_block`) at
  `compare` / `verify-gain` / `keep` / accept that refuses anything not
  `source='real_vllm' AND outcome_class='eval_result'`. A `local_smoke` run can only conclude DoD-B.

## 4. How "better" is decided

Headline metric is **goodput** / `output_throughput_tok_s` (SLO-meeting completions) across three
regimes (`config/bench/profiles/`: throughput / latency / replay). Multi-seed median fitness; too-
noisy runs are `high_variance_inconclusive` (a distinct third state). Acceptance vs the **strong
baseline** requires ALL of: effective (out-of-process probe) + measured quality non-regression +
paired-bootstrap `point ≥ X%` AND `CI_low ≥ X%` + a frozen holdout + `same_caliber`. The headline
verdict and the per-eval result conform to `schemas/eval_result.schema.json`.

## 5. Trust / safety

`src/vllm_evolve/trust/safety.py` (used by the verify path) runs L1 AST safety + signature checks on
candidate policy code; `bench/runtime.py` adds the runtime marker / forgery checks. Marker-forgery
rejection and the out-of-process effectiveness probe (`bench/probe.py`) ensure a candidate cannot
fake "it ran" or "it improved".

## 6. Archive

`ve keep` archives a kept policy + its (real) eval_result under `archive_policies/<target>/<run_id>/`
and records it in the store; `ve discard` appends an audited row to
`archive_policies/<target>/discarded.jsonl` (policy hash + reason + timestamp). LOCK D forbids
archiving a non-real eval_result as a kept winner.

## 7. Layout

| Path | What |
|------|------|
| `src/vllm_evolve/cli/main.py` | the `ve` CLI (all verbs) |
| `src/vllm_evolve/tools/phase_guard.py` | the phase machine |
| `src/vllm_evolve/bench/` | real-vLLM bench + `local_smoke` + config/metrics/slo/compare/decision |
| `src/vllm_evolve/engine/orchestrate.py` | the L1→L4 AutoPT optimization loop |
| `src/vllm_evolve/engine/authoring.py` | Codex/command author contract and manifest import/export |
| `src/vllm_evolve/knowledge/` | research providers, compiler, schemas, and reviewed corpora |
| `src/vllm_evolve/trust/safety.py` | AST safety / signature checks |
| `src/vllm_evolve/install/` | `ve init` installer + PreToolUse hook |
| `src/vllm_evolve/assets/claude/agents/` | the `ve-*` sub-agent definitions (least-privilege scopes) |
| `targets/scheduling/` | the supported target (skeleton + seed + `plugin_template.py`) |
| `config/bench/` | profiles + slo + decision policy |
| `schemas/autopt/` | the sub-agent I/O contracts (Spec / Diagnosis / Profile) |
| `schemas/eval_result.schema.json` | the archived result contract |

## 8. The author intelligence layer (intelligence vs the frozen core)

`ve autopt` supports Claude sub-agents and a Codex-native command author. Both consume the same
`AuthorContext` and remain untrusted proposal producers. Codex assets install the explicit
`ve-context → ve-research → ve-design → ve-generate → ve-evolve` workflow; no Claude-only runtime
is required.

The bright line: **sub-agents are untrusted producers — every output is a PROPOSAL, never a
measurement or a verdict.** The measurement + judgment core (`bench/` native/dispatch/metrics/slo/
compare/decision + `autopt/accept.py` / `verify.py`) is **frozen and agent-free**: a guard test
(`tests/test_frozen_core.py`) walks its transitive import closure and proves no LLM/agent library is
imported anywhere it reaches. The boundary is enforced structurally:

- **Tool scoping** — the interpreting agents (`ve-goal` / `ve-research` / `ve-diagnose`) get no
  `Bash` and no `Write`, so they structurally cannot bench, keep, or author.
- **The author bridge** (`engine/authoring.py`) — a Codex/command author returns policy source plus
  a `CandidateManifest`. The orchestrator checks the parent SHA, research snapshot SHA, mechanism
  IDs, citations, generation, and proposal-only status before writing the candidate. `marker_verified`
  still comes solely from the deterministic bench, so an author can never self-promote.
- **Cited facts** — `ve inspect --verify-citations` checks every `policy_id` a research summary cites
  against the store, so a fabricated citation is detected.
- **The accept threshold** is a config/CLI decision (`run_autopt(accept_threshold_pct=…)`), never the
  agent's `Spec` — a goal can never lower the adoption bar.

Real-box adoption of an authored win (AC7) and a full orchestrated real win (AC8) are `@requires_gpu`
(skipped off-box, never faked); the served-model quality measure needed for code-win adoption is
box-gated future work.

### 8.x Evolution Engine v2 (generational search; search-only)

`engine/evolve_loop.run_evolution` evolves `schedule_batch` over generations: every child is
authored from an `AuthorContext` (diagnosis + parent code/scores + peers + knowledge-store lessons
+ the child's own verify errors for repair + a frozen research snapshot), selection keeps elites
with SHA dedup, the eval budget is charged at every `eval_fn` call (hard cap 24), and lessons are
persisted as IMMUTABLE evidence rows only at run end. Both author forms (deterministic
`template_author_fn` and Codex/command authors) consume the same `to_prompt()` rendering. The
orchestrator holds the pen and stores a manifest next to every candidate. The winner maps back to a
`Candidate` whose `marker_verified` comes from the bench profile — False under any simulator — so
an evolved winner is a PROPOSAL and the adoption gate is untouched (sim = search, real vLLM =
proof).

### 8.y Auto Research Expert Layer

`knowledge/compiler.py` turns a concrete diagnosis, code target, objective, workload, SLO, and
repository revision into a bounded `ResearchContext`:

1. `QueryPlanner` builds target-aware queries and rejects unrelated optimization dimensions.
2. Providers return normalized `ResearchSource` records. The reviewed corpus is always available;
   the optional arXiv provider adds primary-source metadata and records timeout/error outcomes
   instead of making research a hidden availability dependency.
3. The compiler deduplicates by durable identifiers, balances classic and recent evidence, and
   keeps external sources separate from internal run evidence. Falsified hypotheses are retained.
4. `MechanismCard` records the mechanism, applicability constraints, concrete local code hooks,
   expected metric movement, confounders, falsification test, and a bounded implementation recipe.
   Incompatible mechanisms remain visible but are excluded from the active portfolio.
5. The compiler seals the canonical JSON payload with a SHA-256 snapshot hash and writes
   `research_query.json`, `sources.json`, `source_manifest.json`, `mechanism_cards.json`,
   `expert_brief.json`, and `research_snapshot.json`.
6. `evolve_loop` injects that exact snapshot into every child `AuthorContext`; a
   `CandidateManifest` binds each authored source to its parent, snapshot, mechanisms, and sources.

The six generated research artifacts are provenance, not evaluation. Source authority does not
convert a mechanism into a measured win. Only deterministic verification, Frontier search/ablation,
and finally the unchanged real-vLLM acceptance gate can establish increasing levels of evidence.

## 9. The research harness (open-world hypotheses; the frozen core unchanged)

The autopt layer optimizes within a closed set of known options. The **research harness** lets a free
hypothesis be *expressed*, *measured*, and *adjudicated* — turning "is my new idea true?" into a
deterministic, cited verdict — without touching the safety architecture (§5 stays byte-for-byte; the
`test_frozen_core` closure is unchanged; ledger/experiment data is PROPOSAL-layer and never reaches
the adoption gate). Five primitives:

- **P1 experiment-as-data** (`engine/experiment.py`). An `ExperimentSpec` is declarative: a workload
  (a *trace* — the primitive; named shapes like `bursty_multi_tenant` are template generators, the
  guardrail is a **range** check, never an enum of allowed shapes), A/B `arms`, `MetricExpr` metrics,
  `seeds`, and a `budget` (a hard arm×seed cap **charged at the eval-call boundary** — exhaustion →
  `budget_exhausted`). `run_experiment` reduces each metric across seeds by **median**, and only
  yields a number when every spec seed produced a value (a budget-skipped or missing seed → `None`).
- **P2 measurement catalog** (`bench/frontier_catalog.py`, bench layer — does NOT import core/engine,
  so the frozen closure guards it automatically). The whole Frontier output is a queryable catalog;
  a **single** evaluator `evaluate(MetricExpr, catalog)` is shared by P1's metric point-orders and
  P3's adjudication, so a referenced metric is always computable or **explicitly missing** (`None` +
  a missing-columns list — never a fabricated value). `MetricExpr = {column, agg(p50|p90|p99|mean|
  max|count|rate), group_by?, group_reduce?(max|mean|min)}`.
- **P3 event-sourced ledger** (`store/migrations.py` `0006_hypotheses` + `0007` append-only triggers;
  `store/db.py`). Three **insert-only** event rows — `prediction_registered` → `experiment_executed`
  → `adjudication_recorded` — chained by foreign keys (prediction-FIRST is structural; one prediction
  per hypothesis; the adjudication binds to its exact execution event). `BEFORE UPDATE/DELETE`
  triggers `RAISE(ABORT)` so a cited event is immutable even against raw SQL. `verify_citations`
  accepts `hypothesis:<event_id>`; `research.gather()` merges the ledger (incl. **falsified** —
  negative results are first-class, so the loop won't re-propose a refuted idea).
- **P3R deterministic adjudication** (`engine/adjudicate.py`). `adjudicate(prediction, per_arm)` is a
  pure function — inputs are ONLY the prediction + the per-arm cross-seed medians, **no agent
  channel**. Within margin, or any required value missing/malformed → `inconclusive`. Same inputs →
  same verdict, always.
- **P4 free execution channel** (`ve experiment register/run/list`, an admin verb;
  `engine/experiment_chain.run_hypothesis`). `ve-diagnose` may emit a structured
  `{statement, prediction, experiment_spec}` PROPOSAL; the orchestrator routes it to `ve experiment
  run`, which registers → runs → adjudicates → appends the ledger. The agent proposes; the
  deterministic adjudicator decides.
- **P5 bridge-v2 admission throttle** (`integrations/frontier/`). `ve_policy_api.ScheduleDecision`
  adds an optional `defer_ids` field (delivered as a **sanctioned, dependency-free, Frontier-
  importable** surface — a strict superset of the hook-protected, frozen `targets/scheduling/
  skeleton.py` decision, which is NOT mutated; see the Round-6 Plan Evolution). The bridge keeps
  **two strictly separated channels**: a *forgotten* id (never mentioned) is FCFS-appended behind the
  chosen ids (never starves, = v1); a *deferred* id (in `defer_ids`) is **omitted** from the view so
  the policy throttles admission. Anti-starvation decay: an id deferred for ≥ **N = 8** consecutive
  scheduling events is **forced to the queue head** and its counter resets (`ve_defer.DeferTracker`,
  pure single-source logic; the bridge always delegates to it, even on empty defers, so streaks are
  cleared honestly). The `ve_policy_marker.json` records `{invocations, fallbacks, defers}` so the
  caller can prove the policy + defer channel actually ran (a fallback-to-FCFS is never scored as the
  candidate).

The acceptance anchor **H*** ("bursty multi-tenant load: prefix-cache thrash drives the TTFT tail;
grouping admission by tenant improves p99 TTFT") traverses the whole chain on **real CPU Frontier**:
synthesize a bursty multi-tenant trace → register the prediction → A=`vllm_v1` (FCFS) vs
B=`tests/fixtures/policy_group_admission.py` (group by `session_id`, defer non-current tenants) →
grouped p99 TTFT via `MetricExpr` → adjudicate → ledger → `research.gather()` re-cites it. Frontier is
driven out-of-process via the native trace flag `--trace_request_generator_config_trace_file` (the
TRACE_REPLAY generator's config class is `TraceRequestGeneratorConfig`). The pass criterion is **loop
closure, not hypothesis truth** — `supported`/`falsified`/`inconclusive` all pass when the run has
real measurements, real citations, and the B mechanism is proven via the marker; all DoD-B. The real
run is Frontier-gated (skipped off-box, never faked) — see `docs/restructure/MODE_E2E_MATRIX.md`
Phase F.
