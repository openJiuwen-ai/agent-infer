# Internal Design Documents

User documentation lives under `docs/`; development designs live under the repository-root `design/` directory. This
page makes those designs discoverable, but they are not user procedures or stable public API commitments.

## Benchmark Subsystem

- [Benchmark subsystem overview](../../../design/module/benchmarking/index.md)
- [BenchKit orchestration](../../../design/module/benchmarking/benchkit-orchestration.md)
- [Agent runtime adapters](../../../design/module/benchmarking/agent-runtime-adapters.md)
- [Request proxy and hint status](../../../design/module/benchmarking/request-proxy-and-hints.md)
- [Artifacts and evaluation boundary](../../../design/module/benchmarking/artifacts-and-evaluation.md)

These documents record module boundaries, validation paths, and `BENCH-INV-*` development invariants. Use their
status and `last_reviewed` metadata to decide whether they apply to current implementation review.

## AgentCache Skills

- [Skills design](../../../design/superpowers/specs/agentcache-skills-design.md)
- [Skills implementation plan](../../../design/superpowers/plans/agentcache-skills-implementation.md)

The current usage entry point remains the repository-root [`skills/`](../../../skills/README.md). Designs explain
historical decisions and do not replace instructions in each current `SKILL.md`.

## Maintenance Records

- [CI and unit-test fixes](../../../design/ci-ut-fixes.md)
