# Agent Instructions for AgentInfer

These instructions apply to AI-assisted contributions to `agent-infer`,
which publishes the `agentinfer` Python package. Keep this file limited to stable,
repo-wide norms. Evolving RFC workflow, benchmark environment facts, and reviewer
criteria belong in `skills/`, `skills/ac-review/criteria/`, specs, handoff docs,
or benchmark artifacts.

## Development setup

Use the repository's Python and pre-commit configuration as the source of truth.
A typical local setup is:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
pre-commit install
```

## Validation

Before submitting changes, run the relevant local checks for the touched files.
For broad changes, prefer:

```bash
pytest tests/
pre-commit run --all-files
```

For focused changes, run the smallest relevant pytest/pre-commit command and
include the exact command plus result in the PR.

## Commit and PR hygiene

- Every commit must carry a `Signed-off-by:` trailer for DCO compliance.
- Do not include agent co-author attribution in commit messages.
- Fill out `.github/PULL_REQUEST_TEMPLATE.md` with purpose, test plan, and test result.
- Link the relevant issue or RFC when the change affects design, behavior,
  public API, benchmarks, or developer workflow.
- Do not push, force-push, or rewrite shared history unless explicitly requested.

## Documentation and benchmark evidence

- Preserve dated, historical, or archival docs unless the user explicitly asks to rewrite them.
- Prefer adding append-only notes or handoffs over rewriting historical material.
- Do not freeze benchmark host, model, environment, path, or matrix assumptions in
  repo-level agent guidance; cite current artifacts, configs, logs, or handoff
  docs instead.
- Benchmark and performance claims must be backed by reviewer-readable commands,
  raw artifacts/logs, parsed summaries, and an explicit not-run explanation when
  full validation was not run.
