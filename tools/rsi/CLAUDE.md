# vllm-evolve — agent guide

vllm-evolve is a Claude Code–native harness for discovering better **vLLM serving policies**
by evolving policy code and proving it better on **real vLLM** (never a simulator).

## Ground rules

- **Real vLLM only.** The DES simulator was deleted. All evaluation goes through
  `ve bench` (real vLLM in a pinned Docker image). Never reintroduce a simulator and never
  fabricate metrics — the actual run is GPU-gated and `@requires_gpu` tests skip off-hardware
  rather than faking a pass.
- **One supported target: `scheduling`.** `targets/scheduling/` is the only target directory
  shipped; the round verbs (`ve context`/`verify`/`bench`) reject any other target name.
- **Phase-locked rounds.** Work flows through the `ve` CLI under a phase machine
  (`tools/phase_guard.py`) enforced by a PreToolUse hook. Never edit seeds, skeletons,
  `config/`, `schemas/`, or `archive_policies/` — the hook blocks it.
- **UTF-8 everywhere.** This repo is developed on a GBK-default Windows host; every file read
  must pass `encoding="utf-8"` (config/skill/yaml loads silently corrupt otherwise).

## The round (the `ve` flow)

```
ve phase set READ_CONTEXT && ve context scheduling      # gather skeleton/seed/hints/best
ve phase set DESIGN       && ve design --note "..."      # one testable hypothesis
ve phase set GENERATE     ; edit targets/scheduling/work.py   # only here may policy files change
ve phase set VERIFY       && ve verify work.py scheduling     # L1 AST + L2 signature → VERIFY_PASSED
ve phase set BENCHMARK    && ve bench work.py --profile throughput   # real vLLM (GPU)
ve phase set KEEP_OR_DISCARD   && ve compare seed.json work.json     # better/worse/inconclusive
ve phase set COMMIT_OR_ROLLBACK&& ve keep work.py … | ve discard work.py --reason …
```

`ve init` installs the agent + `ve-*` skills + the hook into a project's `.claude/`
(deep-merges settings.json; idempotent; `--uninstall` restores exactly).

## How "better" is decided

Headline metric is **goodput** (requests that complete AND meet the SLO). Three regimes
(`config/bench/profiles/`): throughput (saturating), latency (low-load tight SLO), replay
(real BurstGPT/Azure/ShareGPT/Mooncake traces). Multi-seed median fitness; too-noisy runs
are `high_variance_inconclusive` (a distinct third state). `config/bench/decision.yaml` is
where a human says which profiles must improve / must not regress.

## The autopt sub-agent layer (untrusted intelligence; frozen core)

`ve autopt` has an optional Claude Code sub-agent layer — `ve-goal` / `ve-research` / `ve-diagnose` /
`ve-author`, orchestrated by the `ve-autopt` skill (contracts in `schemas/autopt/`). **Sub-agents are
untrusted producers: every output is a PROPOSAL, never a measurement or a verdict.** The
measurement + judgment core stays **frozen and agent-free** — `tests/test_frozen_core.py` walks its
transitive import closure and proves no LLM/agent import reaches it. Enforced structurally: the
interpreting agents have no `Bash`/`Write` (can't bench/keep/author); the `ve-author` bridge
(`autopt/evolve_target.py:author_evolve_fn`) takes `marker_verified` only from the real bench, so an
author can't self-promote (off-box `local_smoke` → never adopts); `ve inspect --verify-citations`
detects fabricated citations; the accept threshold is a config/CLI decision, never the agent's `Spec`.
Evolution Engine v2: `ve autopt --evolve --backend frontier_sim` runs a GENERATIONAL search (engine/evolve_loop.py — Archive + parent-score feedback via AuthorContext.to_prompt() + repair + a 24-eval hard budget at the eval boundary + run-end lessons in the store); sim scores rank variants only — adoption still requires real vLLM. Real-box behavior (AC7: an authored policy verified on real vLLM; AC8: the orchestrated loop on real
vLLM) is `@requires_gpu` — skipped off-box, never faked. Full code-win **adoption** additionally
needs the served-model quality measure, which is box-gated future work. See **DESIGN.md §8**.

## The research harness (open-world hypotheses; same frozen core)

The autopt loop answers "which known option is best." The **research harness** answers "is my new
hypothesis true." It adds five open primitives on top of the **unchanged** safety architecture
(frozen judgment core / four quarantine locks / `real_source_block` / marker guard / L1L2 verify):
experiment-as-data, a queryable measurement catalog, an event-sourced hypothesis ledger, a free
execution channel, and the bridge-v2 admission-throttling surface. See **DESIGN.md §9**.

```
ve experiment register <spec.json>     # register a prediction (prediction-FIRST, immutable)
ve experiment run <spec.json>          # register → run A/B experiment → adjudicate → append ledger
ve experiment list [--verdict ...]     # the hypothesis ledger (supported / falsified / inconclusive)
```

- **`ve experiment` is an admin verb** (`register_verb("experiment", None)`): the hook allows it in
  every phase, the phase machine is unchanged.
- **A `Spec`** declares a workload (a *trace* — the primitive; `bursty_multi_tenant` etc. are just
  template generators, the guardrail is a **range**, never an enum), A/B `arms` (a policy + knob
  delta), structured `MetricExpr` metrics (`{column, agg(p50|p90|p99|mean|max|count|rate), group_by?,
  group_reduce?}`), `seeds`, and a `budget` (a hard arm×seed cap charged at the eval-call boundary).
- **The verdict is deterministic** (`engine/adjudicate.py`, a pure function of the prediction + the
  per-arm cross-seed **median**): within margin or any required value missing → `inconclusive`. A
  metric the catalog cannot compute is explicit `None` — **never fabricated**.
- **The ledger is append-only** (`hypothesis_events`, migration `0006` + `0007` `BEFORE UPDATE/DELETE`
  triggers): `prediction_registered → experiment_executed → adjudication_recorded`, chained by
  foreign keys; cited event ids never change. `ve inspect --verify-citations` validates
  `hypothesis:<event_id>` citations; `research.gather()` surfaces them — **`falsified` is first-class
  memory** (it stops the loop re-proposing a refuted idea).
- **PROPOSAL-layer, never adoption.** Experiment/ledger evidence (any verdict) never feeds the
  adoption gate — a code/config win still requires the real-vLLM gate. The plan's acceptance anchor
  is **loop CLOSURE, not hypothesis truth**: `supported` / `falsified` / `inconclusive` all pass when
  the loop is real, cited, and DoD-B. The H* e2e runs on real CPU Frontier (`@requires`-Frontier,
  skipped off-box) — see **`docs/restructure/MODE_E2E_MATRIX.md` Phase F**.

## Layout

| Path | What |
|------|------|
| `src/vllm_evolve/cli/main.py` | the `ve` CLI (all verbs) |
| `src/vllm_evolve/bench/` | real-vLLM benchmark: metrics, slo, compare, decision, runner, backend, datasets, evaluator |
| `src/vllm_evolve/install/` | `ve init` installer + PreToolUse hook decision logic |
| `src/vllm_evolve/assets/claude/` | shipped agent + skills + hook + settings fragment |
| `src/vllm_evolve/tools/phase_guard.py` | the phase machine |
| `src/vllm_evolve/engine/` | research harness: `experiment.py` (spec+runner), `adjudicate.py` (pure verdict), `experiment_chain.py` (register→run→adjudicate→ledger + Frontier eval) |
| `src/vllm_evolve/bench/frontier_catalog.py` | the measurement catalog + single `MetricExpr` evaluator |
| `src/vllm_evolve/store/` | the hypothesis ledger (`migrations.py` 0006/0007, `db.py`) |
| `integrations/frontier/` | the ve_policy bridge patch (`ve_policy.patch`), the pure defer decay (`ve_defer.py`), the policy-facing defer API (`ve_policy_api.py`) |
| `targets/scheduling/` | the supported target (skeleton + seed + `plugin_template.py`) |
| `config/bench/` | profiles + slo + decision policy |
| `schemas/eval_result.schema.json` | the archived result contract |

## Dev setup (do this first)

```bash
pip install -e .                     # editable install — REQUIRED
python -c "import vllm_evolve, pathlib; p=pathlib.Path(vllm_evolve.__file__).resolve(); assert p.is_relative_to(pathlib.Path('src/vllm_evolve').resolve()), p"
```

**Shadow trap:** if a sibling checkout was editable-installed first, `import vllm_evolve` silently
resolves to *that* tree and `ve`/pytest run the wrong code. The assert above is folder-name-agnostic:
run it from THIS checkout's root — it confirms the import resolves under *this* `src/`, not a sibling.
Re-run `pip install -e .` from here if it fails.

## Run the whole flow on a laptop (no GPU) — `local_smoke`

The entire `ve` round + a full `ve autopt` loop run end-to-end off-box with an opt-in in-process
backend, so you can exercise the plumbing without a GPU:

```bash
ve bench work.py --runner candidate --backend local_smoke   # or: export VE_BENCH_BACKEND=local_smoke
ve autopt "maximize throughput" --backend local_smoke
```

**`local_smoke` is SYNTHETIC plumbing only — its numbers are not real and can NEVER become a
gain / keep / AC6 result.** It is quarantined by four independent locks: `source='local_smoke'`,
a failure-class `outcome_class`, `effective=False`/`quality_ok=None` (never the plugin marker), and
a real-source guard at `compare`/`verify-gain`/`keep`/accept that hard-refuses any non-`real_vllm`
input (`non_real_source_blocked`). A `local_smoke` run can only ever conclude **DoD-B**. Real
numbers require the box (`--backend remote`, the default) — see `docs/AC6_RUNBOOK.md`.

A second non-real backend, **`frontier_sim`**, drives the Frontier discrete-event simulator as a
local subprocess (`VE_FRONTIER_REPO`/`VE_FRONTIER_PYTHON`; never a Python dep) and — via the
`integrations/frontier/` ve_policy bridge patch — **genuinely executes evolved `schedule_batch`
policies for evolutionary SEARCH**: `ve autopt --evolve --backend frontier_sim` ranks variants by
sim score into a sim-winner PROPOSAL (`runs/sim_winner/`). Sim scores are search-only; adoption
still requires real vLLM, and frontier_sim sits behind the same four quarantine locks (DoD-B only).
See `docs/restructure/FRONTIER_EVOLVE_DESIGN.md`.

## Running checks

```bash
python -m pytest -q                  # full suite (GPU tests skip off-hardware)
python -m pytest tests/bench -q      # the benchmark core
ruff check src tests                 # lint (must be clean)
```

**Windows note:** `pytest -q` may print a trailing `PermissionError: [WinError 5] ...
pytest-current` from pytest's own temp-symlink cleanup. It fires AFTER the run, does not affect the
result (exit code stays 0, the `N passed` line precedes it); add `-p no:cacheprovider` to suppress it.

Removed sim-based tests and their real-vLLM analogues are tombstoned in
`tests/REMOVED_SIM_TESTS.md`.
