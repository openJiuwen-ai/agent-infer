# AgentInfer Skills

ZCode/agent skills for working on AgentInfer's AgentCache and scheduling stack. Each skill lives in its own
directory as a `SKILL.md` file and is invoked by an AI agent via the Skill
tool.

## Skills

| Skill | Audience | Purpose |
| --- | --- | --- |
| `ac-bootstrap` | Contributor | Scaffold a cache integration, scheduling component, or entry point. |
| `ac-benchmark` | Contributor + Integrator | Run the benchmark harness vs vLLM, capture and compare metrics. |
| `ac-integrate` | Integrator | Plug AgentInfer AgentCache into a vLLM serving deployment. |
| `ac-review` | Contributor | Review changes against repo conventions that grow over time. |
| `ac-design` | Contributor | Analyze cache patterns and write design docs for optimization strategies. |
| `agentbench-investigate-regression` | Contributor + Integrator | Diagnose AgentBench regressions from finalized run, task, session, and service evidence. |

All producer skills (`ac-bootstrap`, `ac-benchmark`, `ac-integrate`,
`ac-design`) end by invoking `ac-review` as the final pre-merge validation
step. `agentbench-investigate-regression` likewise ends at its report and is
read-only; code or harness changes motivated by its findings are validated
with `ac-review` separately.

## File Conventions

- Each skill directory contains a `SKILL.md` with `name` and `description`
  frontmatter. The `description` is the auto-discovery trigger — it must let an
  agent decide "yes, invoke this" from the user's request alone.
- `ac-review` stores its evolving criteria in `criteria/*.md`; see
  `ac-review/criteria/README.md` for the weekly refresh protocol.
