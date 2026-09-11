# AgentCache Skills

ZCode/agent skills for working on AgentCache. Each skill lives in its own
directory as a `SKILL.md` file and is invoked by an AI agent via the Skill tool.

## Skills

- `ac-bootstrap` (Contributor) — Scaffold a new cache backend, eviction policy,
  or entry point.
- `ac-benchmark` (Contributor + Integrator) — Run the benchmark harness vs
  vLLM, capture and compare metrics.
- `ac-integrate` (Integrator) — Plug AgentCache into a vLLM serving deployment.
- `ac-review` (Contributor) — Review changes against repo conventions that grow
  over time.

All three producer skills (`ac-bootstrap`, `ac-benchmark`, `ac-integrate`) end
by invoking `ac-review` as the final pre-merge validation step.

- `agentbench-integrate-agent` — Integrate and validate coding-agent runtimes against common benchmark contracts.
- `agentbench-investigate-regression` — Diagnose performance changes from finalized run artifacts.

## Conventions

- Each skill directory contains a `SKILL.md` with `name` and `description`
  frontmatter. The `description` is the auto-discovery trigger — it must let an
  agent decide "yes, invoke this" from the user's request alone.
- `ac-review` stores its evolving criteria in `criteria/*.md`; see
  `ac-review/criteria/README.md` for the weekly refresh protocol.
