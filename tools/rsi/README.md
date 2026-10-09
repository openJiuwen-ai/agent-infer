# vllm-evolve

![vllm-evolve](docs/assets/vllm-evolve-logo.svg)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Agent-authored serving policies, measured by reproducible experiments.

---

## What is vllm-evolve?

vllm-evolve evolves vLLM scheduling policy source, verifies every candidate, measures it, and
retains complete lineage and evidence. Claude Code and Codex integrations share the same `ve` CLI,
phase guard, author contract, evolution engine, and frozen adoption boundary.

There are two deliberately separate evaluation paths:

- **Local Frontier search** runs the real CPU Frontier simulator as an out-of-process
  `python -m frontier.main` process. The patched `ve_policy` scheduler actually calls each
  candidate's `schedule_batch`, so policy code changes can change the result. Its output is always
  `source=frontier_sim`, `outcome_class=simulator_nonqualifying`, and a `sim_winner`; it cannot be
  passed to production `keep`.
- **Real vLLM validation** runs the candidate on the GPU/vLLM benchmark path. Only clean
  `source=real_vllm` evidence may cross the production adoption boundary.

`local_smoke` remains a synthetic plumbing test and is never evidence of algorithm quality.

Only the `scheduling` target is shipped. The legacy in-process DES implementation is gone; Frontier
is a separate, pinned local checkout rather than an embedded simulator.

## Quick start

Run all commands below in **Linux or WSL 2**, using Python 3.12 and Git. Installation and
local simulation need no GPU, model weights, SSH access, or API key. Real-vLLM validation
is a separate step that needs its own hardware and runtime.

This directory is a standalone toolkit: installing AgentInfer at the repository root does
not install the `ve` command. Keep the RSI environment separate from your serving environment.

### 1. Install and check the CLI

```bash
git clone https://github.com/openJiuwen-ai/agent-infer.git
cd agent-infer/tools/rsi

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ve --help

# Materialize .codex agent/hooks and .agents/skills from packaged assets.
ve init --client codex
ve init --client codex --check
```

Already cloned AgentInfer? Start at `cd tools/rsi` from its root. Keep this directory as
your working directory for the remaining commands; `targets/`, `config/`, and `runs/` are
resolved here. `--check` verifies that the generated client files match the packaged assets.

### 2. Prepare the simulator and workload

Frontier runs in its own Python environment. The commands below create that environment
explicitly; both external repositories are pinned for reproducibility.

```bash
# Still in agent-infer/tools/rsi, with the RSI environment active.
git clone https://github.com/NetX-lab/Frontier.git ../Frontier
git -C ../Frontier checkout a4b22df8211864bf229258ecdfbe680f048f2d77
python3.12 -m venv ../Frontier/.venv
../Frontier/.venv/bin/python -m pip install -e ../Frontier
python integrations/frontier/ensure_patch.py ../Frontier

export VE_FRONTIER_REPO="$(cd ../Frontier && pwd)"
export VE_FRONTIER_PYTHON="$VE_FRONTIER_REPO/.venv/bin/python"

# Official BurstGPT data, not tests/bench/fixtures.
git clone https://github.com/HPMLL/BurstGPT.git ../BurstGPT
git -C ../BurstGPT checkout d895a53bb7b8ec137d0d2fe203b335835a78c10a
test -s ../BurstGPT/data/BurstGPT_1.csv
```

The patch command reports `applied` on the first run and `already_applied` on subsequent
runs. It rejects a wrong Frontier commit or a partially applied patch.

### 3. Run a small CPU search and read the result

The first run uses a **deterministic template author**, not an LLM. `ve init --client codex`
installs client integration but does not automatically connect a Codex author. To use one,
provide `--author-command` with an adapter implementing the [author contract](#auto-research-expert-layer).

```bash
ve frontier-evolve \
  --burstgpt ../BurstGPT/data/BurstGPT_1.csv \
  --out runs/frontier_local_evolution \
  --seeds 0,1,2 \
  --fragment-size 32 \
  --slo-ttft-ms 200 \
  --max-num-seqs 4 \
  --generations 2 \
  --population 5 \
  --max-total-evals 10 \
  --research-mode offline
```

`--research-mode offline` uses the bundled research corpus without live metadata requests.
This is a small trial, not a promise of a winning policy. The command exits zero only when
the frozen candidate passes the held-out acceptance and ablation gates. A completed search
with no accepted winner exits nonzero too; check the report before treating that as a setup error.

Start with `runs/frontier_local_evolution/report.md`. An accepted candidate and its evidence
are under `sim_winner/`; a rejected result is under `no_winner/`. Raw commands, traces,
seeds, metrics, policy markers, hashes, and lineage stay in the same run directory.

If startup fails, read the JSON `error` printed by the CLI and check these first:

- `ve: command not found`: reactivate `tools/rsi/.venv` and repeat the editable install.
- Missing Frontier interpreter: verify `"$VE_FRONTIER_PYTHON" --version` and the two exports above.
- Missing trace: check that `../BurstGPT/data/BurstGPT_1.csv` is the real CSV, not a Git LFS pointer.
- Imports of `resource` or `fcntl` fail: use Linux/WSL 2, not native Windows Python.

A `sim_winner` is a simulator proposal, **not a production speedup** and cannot be adopted
with `ve keep`. It needs separate real-vLLM validation.

The completed reference run and its honest limitations are documented in
[`reports/frontier_local_evolution.md`](reports/frontier_local_evolution.md).

For the complete local-Frontier → remote real-vLLM workflow, environment
deployment, two-GPU hard limit, mechanism control, and artifact layout, see
[`docs/REMOTE_REAL_EVOLVE_RUNBOOK.md`](docs/REMOTE_REAL_EVOLVE_RUNBOOK.md).
Remote host names and paths are examples: copy `config/remote/gpu.env.example` to
an untracked `.env` and configure your own SSH alias and writable workspace.

## Codex integration

The authored source of truth lives in `src/vllm_evolve/assets/codex/`; generated `.codex/` and
`.agents/` directories are ignored. `ve init --client codex`:

- installs the `vllm-policy-optimizer` agent;
- installs the `ve-context`, `ve-research`, `ve-design`, `ve-generate`, and `ve-evolve` skills
  under `.agents/skills`;
- installs a phase/protected-path hook under `.codex/hooks`;
- merges a bounded vllm-evolve section into `AGENTS.md`;
- supports idempotent `--check`, `--force`, and `--uninstall`.

Use `ve init --client claude` for Claude Code, or `--client both` for both clients. The Codex author
receives the complete `AuthorContext.to_prompt()` JSON, including diagnosis, parent source and
score, peer archive, failed verification, lessons, generation, remaining budget, and a frozen
`ResearchContext`. It returns a complete source string plus a `CandidateManifest`; the orchestrator
validates the research citations and lineage before it writes candidate files. No Claude sub-agent
or Claude-only tool is needed by the Codex path.

## Auto Research Expert Layer

Before a code target is authored, vllm-evolve can compile an optimization-specific evidence packet
instead of asking a general coding agent to improvise from memory:

```text
Diagnosis + target + workload/SLO
  → target-aware research query
  → reviewed primary-source corpus + optional live arXiv metadata
  → internal lessons and falsified hypotheses (kept in a separate evidence plane)
  → ranked MechanismCards with concrete repository hooks and falsification tests
  → immutable ResearchContext / ExpertBrief
  → AuthorContext
  → source + CandidateManifest
```

The built-in scheduling corpus contains paper-level mechanism descriptions and implementation
maps, not copied paper bodies. `auto` mode attempts a live primary-source refresh and records every
provider outcome; `offline` mode is deterministic. Both modes freeze the exact evidence snapshot
used by a generation so a later refresh cannot silently change an experiment.

```bash
# Build and inspect a scheduling expert packet.
ve research build \
  --target scheduling \
  --goal "maximize goodput on BurstGPT under TTFT SLO" \
  --environment '{"workloads":["BurstGPT"],"vllm_version":"0.21.0"}' \
  --mode auto \
  --out runs/research/scheduling

ve research verify runs/research/scheduling/research_snapshot.json

# Give an external Codex process a self-contained author contract, then validate
# its source + manifest before importing it.
ve research export-author \
  runs/research/scheduling/research_snapshot.json \
  --out runs/research/author_bundle

ve research import-candidate \
  --author-context runs/research/author_bundle/author_prompt.json \
  --source candidate.py \
  --manifest candidate.manifest.json \
  --out runs/research/imported.json
```

`ve autopt --evolve`, `ve run --evolve`, and `ve frontier-evolve` accept
`--research-mode`, `--research-snapshot`, and `--research-refresh`. A configured
`--author-command` receives one `AuthorContext` JSON document on stdin and must return
`{"source": "...", "manifest": {...}}` on stdout. If no external author is configured, the
deterministic template author still consumes the same mechanism cards, making offline tests
reproducible without pretending that the template is Codex.

Research evidence only guides proposal generation. It cannot set benchmark metrics, claim a gain,
or bypass verification. Frontier results remain `simulator_nonqualifying`; only the existing real
vLLM gate can adopt a production winner.

## Evolution and acceptance

The local workflow is:

```text
research/context
  → diagnosis
  → frozen ResearchContext / ExpertBrief
  → AuthorContext.to_prompt()
  → source + CandidateManifest author response
  → L1/L2 verification
  → real Frontier subprocess + ve_policy marker
  → generation archive / parent feedback
  → frozen winner
  → once-only held-out baseline/candidate/ablation audit
  → sim_winner + report
```

Search sees chronological BurstGPT train/validation fragments and separately labeled synthetic
stress workloads. BurstGPT has no tenant or prefix information, so conversion uses one neutral
session and empty prefix hashes. The test fragment is not materialized until after the source SHA is
frozen.

Every held-out scenario uses paired seeds `0,1,2`. The opponent is the strongest per-scenario
policy among FCFS, SJF, LJF, LIFO, and the current scheduling seed. Acceptance requires:

- median goodput gain at least 3%;
- positive gain on at least two of three held-out scenarios;
- no scenario below -2%;
- no completion loss, a valid policy marker, and no all-fallback execution;
- an ablation supporting the new mechanism.

There is intentionally no automated novelty score or novelty gate. Algorithmic value is reviewed
from its control flow, pseudocode, source diff, failure history, and ablation.

## Phase-locked real-vLLM round

The original production workflow remains available:

| Step | Command | Phase |
| --- | --- | --- |
| Read context | `ve context <target>` | READ_CONTEXT |
| Design | `ve design --note "<hypothesis>"` | DESIGN |
| Edit policy | edit `targets/<target>/*.py` | GENERATE |
| Verify | `ve verify <policy> <target>` | VERIFY |
| Benchmark | `ve bench <policy> --profile <profile>` | BENCHMARK |
| Compare | `ve compare <baseline.json> <candidate.json>` | KEEP_OR_DISCARD |
| Decide | `ve keep …` / `ve discard …` | COMMIT_OR_ROLLBACK |

The hook blocks out-of-order actions and edits to frozen seeds, skeletons, configs, schemas, and
archive provenance. Production adoption still requires real-vLLM evidence.

## Architecture

```text
src/vllm_evolve/
├── cli/main.py                 ve CLI, including init and frontier-evolve
├── knowledge/
│   ├── providers.py            curated/offline and live primary-source providers
│   ├── compiler.py             query, ranking, implementation map, frozen snapshots
│   ├── schemas.py              ResearchSource/Card/Context/CandidateManifest contracts
│   └── corpora/                reviewed scheduling metadata and mechanism cards
├── engine/
│   ├── authoring.py            external author/export/import manifest boundary
│   ├── evolve_loop.py          bounded generational archive/lineage loop
│   ├── evolve_target.py        source-returning author and structural grammar
│   └── local_frontier_evolve.py
│                                data split, baselines, search, held-out and ablation
├── bench/
│   ├── frontier_sim.py         real local Frontier subprocess, quarantined evidence
│   ├── datasets/burstgpt.py    streaming 6/8-column loader and deterministic conversion
│   └── native.py               real-vLLM benchmark path
├── install/                    Claude/Codex installers and shared hook decision
├── assets/{claude,codex}/      packaged agent, skill, hook and guide sources
├── core/                       frozen verification and adoption rules
└── trust/ store/               safety checks, provenance and SQLite evidence
```

## Verification

From `tools/rsi/` in the RSI environment:

```bash
python -m pytest -q
ruff check src tests integrations/frontier/*.py

# Optional real-Frontier tests run only when VE_FRONTIER_* is configured.
VE_BURSTGPT_CSV=../BurstGPT/data/BurstGPT_1.csv \
  python -m pytest -q tests/test_frontier_local_evolution_real_e2e.py
```

The repository-level [RSI workflow](../../.github/workflows/rsi.yml) runs CPU tests on
Python 3.10/3.12 and blocks on ruff errors in maintained code. GPU and real-Frontier
tests skip unless their prerequisites are configured. Parent pre-commit auto-formatters
exclude this imported tree to preserve frozen policy bytes and historical evidence;
RSI uses its own lint configuration. This CI does not certify GPU performance.

## License

Apache 2.0 — see [LICENSE](LICENSE).
