# RSI component implementation boundaries

The initial scaffold implements controller state, feedback contracts/aggregation, scoped knowledge, storage, CLI and a
local demo UI.
The following directories are an implementation plan, not existing adapters. Add them when their first real
integration is built;
do not populate the repository with empty modules that imply working hardware support.

| Planned directory | Files / responsibility | First integration contract |
| --- | --- | --- |
| `harness/` | `runtime.py`, `dag.py`, `context.py`, `tools.py`, `sandbox.py`, `lifecycle.py` | Role sessions, DAG dependencies, exact knowledge snapshot, tool permissions and cancellation |
| `agents/` | `planner.py`, `profiler.py`, `implementer.py`, `reviewer.py`, `contracts.py` | Structured hypothesis, profile interpretation, patch and independent review artifacts |
| `adapters/zcode/` | `client.py`, `session.py`, `artifact_adapter.py` | Pin ZCode revision; implement its verified local CLI/API protocol, not an assumed interface |
| `adapters/semantic_router/` | `snapshot.py`, `policy.py`, `metrics.py` | Export config/policy versions, route decisions and supported lifecycle hints |
| `adapters/router/` | `requests.py`, `scheduler.py`, `metrics.py` | Trace correlation, worker placement, cancellation and session version pinning |
| `adapters/vllm/` | `api_server.py`, `engine.py`, `worker.py`, `model_scripts.py`, `parallel.py`, `ops/communication.py`, `ops/compute.py` | Per-layer read-only probes first, then controlled candidate configuration adapters |
| `adapters/vllm_ascend/` | Same seven logical layers plus `capabilities.py` | Separate CANN/HCCL/model/hardware/version manifest; unavailable capabilities stay explicit |
| `experiments/` | `spec.py`, `runner.py`, `leases.py`, `worktree.py`, `manifest.py`, `reconcile.py` | Freeze variables/criteria; isolate builds and device resources; recover interrupted attempts |
| `feedback/validators/` | `numeric.py`, `protocol.py`, `behavior.py`, `performance.py`, `task.py` | Implement trusted thresholds/statistics, measurement provenance and held-out task checks |
| `evidence/` | `store.py`, `manifest.py`, `integrity.py`, `retention.py` | Content-addressed artifacts, immutable provenance, hashes, retention, redaction |
| `knowledge/curation/` | `extract.py`, `review.py`, `invalidate.py`, `retrieve.py` | Turn evidence into proposals, validate applicability, retain negative outcomes and invalidate stale facts |
| `evaluation/` | `policy.py`, `runner.py`, `holdout.py`, `attestation.py` | Independent frozen gate, protected tasks and verified result attestation |
| `promotion/` | `registry.py`, `activate.py`, `observe.py`, `rollback.py`, `reconcile.py` | Separate accepted/active/last_good, canary/health gates, uncertain-state reconciliation |
| `dashboard/` | Extend `server.py` with authenticated commands and event cursors | Connect UI to one state authority before enabling real interventions |

## Artifact links and identifiers

Use `run_id → round_id → hypothesis_id → candidate_id → trial_id → check_id` throughout.
A candidate manifest records primary and affected layers, repository SHA and patch hash, base image and dependencies,
engine/plugin/model/tokenizer/template revisions, device topology, workload digest and frozen acceptance-policy digest.
Each check points to measured artifacts; a KnowledgeProposal points to both supporting and contradicting trial IDs.
A PromotionRecord points to independently verified acceptance plus actual deployment observations.

The initial `FeedbackRecord` is a smaller implemented contract. Store future rich manifests alongside it before
extending the schema;
do not imply that the bootstrap already attests code identity, hashes or artifact integrity.

## Scheduling and resource isolation

Harness schedules agent tasks. Global Scheduler dispatches inference requests. RSI Controller schedules experiments
and manages versions.
Stable optimization-agent serving and candidate experimental serving use separate processes/resources or explicit time
leases.
Prefer one changed variable per first trial; cross-layer hypotheses list all affected layers and require cross-layer
regression checks.
Do not let an experimental scheduler starve the stable model serving the optimization agents.

## Delivery sequence

1. Current PR: offline state/evidence/knowledge contracts, UI, documentation, images and presentation.
2. Event integration: unify Dashboard and Controller state, add authenticated actor/reason and atomic snapshots/cursors.
3. One backend: pin a known supported model/runtime; connect protocol and numerical checks before timing experiments.
4. Candidate runner: isolated worktrees/builds, leases, evidence integrity and paired unprofiled performance tests.
5. Independent task gates and observed activation/recovery, then multi-agent ZCode execution.
6. Second backend: independently qualify Ascend, retaining shared taxonomy but separate collectors and evidence.

Hardware, model support, performance gains and deployment behavior are deliberately unverified in the first delivery.
