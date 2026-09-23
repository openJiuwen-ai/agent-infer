# Contributing to vllm-evolve

> The authoritative, always-current guide is **[CLAUDE.md](CLAUDE.md)**. This file is the
> human-facing companion: setup, checks, and how to land a change. When the two disagree,
> CLAUDE.md wins — and please open a PR fixing this file.

vllm-evolve discovers better **vLLM serving policies** by evolving policy code and proving it
better on **real vLLM** (never a simulator — the DES sim was removed). Work flows through the
`ve` CLI under a phase machine.

## Setup

```bash
git clone https://github.com/vllm-project/vllm-evolve
cd vllm-evolve
python -m venv .venv && source .venv/bin/activate   # recommended
pip install -e ".[dev]"                             # editable install — REQUIRED
```

**Verify the install resolves to *this* checkout** (the "shadow trap": a sibling checkout that
was editable-installed first will silently shadow this one — e.g. after renaming the repo
directory without re-running `pip install -e .`):

```bash
python -c "import vllm_evolve, pathlib; p=pathlib.Path(vllm_evolve.__file__).resolve(); assert p.is_relative_to(pathlib.Path('src/vllm_evolve').resolve()), p"
```

If that fails, re-run `pip install -e .` from this directory.

## Running checks

```bash
python -m pytest -q                  # full suite (GPU tests skip off-hardware)
python -m pytest tests/bench -q      # the benchmark core
ruff check src tests                 # lint — must be clean
```

`@requires_gpu` tests skip off-box rather than faking a pass; never fabricate metrics. On
Windows, `pytest -q` may print a trailing `PermissionError: [WinError 5] ... pytest-current`
from pytest's own temp cleanup — it fires *after* the run and does not affect the result; add
`-p no:cacheprovider` to suppress it.

## The round (the `ve` flow)

Every change to a policy goes through a phase-locked round, enforced by a PreToolUse hook:

```
ve phase set READ_CONTEXT && ve context scheduling      # gather skeleton/seed/hints/best
ve phase set DESIGN       && ve design --note "..."      # one testable hypothesis
ve phase set GENERATE     ; edit targets/scheduling/work.py   # ONLY here may policy files change
ve phase set VERIFY       && ve verify work.py scheduling     # L1 AST + L2 signature
ve phase set BENCHMARK    && ve bench work.py --profile throughput   # real vLLM (GPU)
ve phase set KEEP_OR_DISCARD    && ve compare seed.json work.json    # better/worse/inconclusive
ve phase set COMMIT_OR_ROLLBACK && ve keep work.py …  |  ve discard work.py --reason …
```

No GPU? Run the whole round (and a full `ve autopt` loop) off-box with the synthetic in-process
backend — useful for exercising the plumbing, but its numbers are **never** real and can never
become a gain/keep:

```bash
ve bench work.py --runner candidate --backend local_smoke
ve autopt "maximize throughput" --backend local_smoke
```

Install the agent + `ve-*` skills + the phase-guard hook into a project (including this repo, to
dogfood) with `ve init`; `ve init --uninstall` restores exactly.

## Improving the scheduling policy

`scheduling` is the **only** supported target. The evolvable surface is the single function
`schedule_batch(...)` — see [`targets/scheduling/skeleton.py`](targets/scheduling/skeleton.py)
for the exact signature, the `RequestInfo` / `ScheduleDecision` dataclasses, and the constraints
(use only the provided request IDs, respect `max_num_batched_tokens` / `max_num_seqs`, no O(n²)
loops). Read [`targets/scheduling/prompt_hints.md`](targets/scheduling/prompt_hints.md) and the
top seeds under `targets/scheduling/seeds/` for strategies that already work.

To propose a change: edit `targets/scheduling/work.py` (in the GENERATE phase only), then
`verify → bench → compare`. If it improves the headline metric without regressing the guardrails
in [`config/bench/decision.yaml`](config/bench/decision.yaml), open a PR with the `eval_result`
JSON from the real-vLLM run.

## How "better" is decided

Headline metric is **goodput** (requests that complete AND meet the SLO), measured across three
regimes (throughput / latency / replay) defined in `config/bench/`. See **CLAUDE.md** and
**[DESIGN.md](DESIGN.md)** for the full decision model — do not hand-copy formulas into code or
docs; import the canonical implementation.

## Ground rules (the hook enforces most of these)

- **Real vLLM only.** Never reintroduce a simulator; never fabricate metrics.
- **Don't edit ground truth.** Seeds, skeletons, `config/`, `schemas/`, `archive_policies/`, the
  phase state, and the installed `.claude/` assets are off-limits — the hook blocks writes to them.
- **One authored home for Claude assets.** The agent / `ve-*` skills / hook live in
  `src/vllm_evolve/assets/claude/`; the repo's own `.claude/` is a generated `ve init` install
  (gitignored) — edit the source, not the install.
- **UTF-8 everywhere.** This repo is developed on GBK-default Windows hosts; always read/write
  files with `encoding="utf-8"`.

## Code style

```bash
ruff check src targets tests
```

Removed sim-era tests and their real-vLLM analogues are tombstoned in
[`tests/REMOVED_SIM_TESTS.md`](tests/REMOVED_SIM_TESTS.md).
