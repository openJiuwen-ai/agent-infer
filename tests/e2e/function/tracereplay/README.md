# Trace replay (functional E2E)

Suites under this directory share the **pattern**: vLLM serve (from case JSON) + agentbench
**replay** bench + optional cold-start repeatability checks.

| Suite | Directory | Bench config | Trace |
| ----- | --------- | ------------ | ----- |
| Inferact Codex SWE-bench Pro | [`inferact/`](inferact/README.md) | `replay_inferact.yaml` | `codex_swebenchpro.json` |

Shared infrastructure: [`../../helpers/`](../../helpers/) (case load, vLLM lifecycle, run validation).

New trace families: add `tracereplay/<suite>/` with its own `cases/`, `run_*.py`, suite-specific
helpers, and link it from the table above.
