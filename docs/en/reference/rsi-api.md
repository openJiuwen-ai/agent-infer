# RSI Python and HTTP API (bootstrap)

Importing `agentinfer.rsi` has no hardware side effects. Run `python -m agentinfer.rsi`, or `agentinfer-rsi` after
package installation.

## Implemented directory map

```text
agentinfer/rsi/
  __init__.py, __main__.py, cli.py      # lightweight CLI and JSON output
  controller/service.py               # ten-stage state, budgets, revision, idempotency
  feedback/
    models.py                         # FeedbackRecord and exact-scope validation
    taxonomy.py                       # seven-layer CUDA/Ascend signal/check map
    probes.py                         # Probe protocol and UnavailableProbe
    engine.py                         # frozen-check evidence aggregation
  knowledge/
    repository.py                     # scoped retrieval, lifecycle and history
    seeds.py                          # drafts for 22 namespaces
  storage/sqlite.py                   # transactional state, events, commands, knowledge
  dashboard/
    server.py                         # loopback demo HTTP API
    static/dashboard.html             # six-view offline prototype
examples/rsi/feedback.json             # synthetic feedback example
tests/rsi/                            # standard-library CPU behavioral tests
tools/rsi/build_presentation.mjs       # slide and engineering-figure source
docs/assets/rsi/                       # PPT, PNG, SVG and Mermaid
```

## FeedbackRecord

Identity fields are check_id, hypothesis_id, candidate_id and baseline_id. backend is cuda/ascend; layer uses the
seven-layer enum.
category is correctness/system_behavior/performance; verdict is pass/fail/inconclusive/unavailable.
metric/value/unit describe the measurement; unavailable requires null value.
scope fixes engine_version, model_revision and workload_id, with optional additional dimensions such as
shape/dtype/topology.
diagnostic indicates measurement perturbation, evidence_refs are provenance pointers, next_test states the next
action, and synthetic marks demonstration data.

```python
from agentinfer.rsi.feedback import evaluate_feedback, list_layers, load_records

layers = list_layers("ascend")
records = load_records("examples/rsi/feedback.json")
# A trusted caller should load scope and required_checks from a frozen contract.
```

`evaluate_feedback(records, required_checks, scope)` returns status/checks/reason/next_test. Evaluation scope
additionally requires
candidate_id/baseline_id/backend. Missing/duplicate/mismatched/unreferenced/synthetic evidence and profiled
performance are INCONCLUSIVE.
A valid FAIL dominates INCONCLUSIVE; all required checks need valid PASS for aggregate PASS.
Upstream validators aggregate repeated trials using frozen statistical rules. This function neither executes probes,
authenticates referenced files nor recomputes thresholds, and cannot deploy production.

## Controller

`Controller(db_path)` exposes create/get/list_runs/events/command.
`create(run_id, baseline=..., max_trials=3, demo=False, backend='cuda')` disables simulated acceptance/deployment by
default.
`command(run_id, expected_revision=..., idempotency_key=..., action=..., payload={})` atomically stores state, audit
and replay response.

| action | Condition / payload |
| --- | --- |
| advance | 1→2→3→4 or 5→6→7 |
| start_trial | Stage 4, consumes one trial; optional candidate_id |
| pause / resume | Pause new work; in-flight demo evaluation, observation and recovery may complete |
| update_budget | max_trials cannot erase spent trials |
| retry | Stage 7 FAIL/INCONCLUSIVE with remaining budget, returns to 3 |
| exclude | Exclude candidate at stage 5/6/7, return to 3 |
| demo_evaluate | Stage 7; result: PASS/FAIL/INCONCLUSIVE |
| demo_activate | Stage 8; outcome: success/failed/unknown |
| demo_soak | Stage 9; result: PASS/FAIL |
| demo_rollback | Stage 8/9; healthy: boolean |
| next_round | Stage 10, no recovery outstanding and remaining budget |
| complete | Stage 10, or archive early at stage 1–4 before starting a trial |

demo_* requires demo=True. Failed recovery blocks other work until recovery is confirmed healthy.
Production runs cannot currently cross the independent acceptance/deployment boundary.

## KnowledgeRepository

`KnowledgeRepository(db_path)` exposes propose/get/search/transition/seed_demo.
`search(scope=..., backend=..., component_version=..., workload=..., text='', status=None)` matches all scope fields
exactly,
excluding stale/retracted by default. Required proposal fields are
namespace/scope/backend/component_version/workload/title/claim;
evidence is a list of ref/outcome objects. transition requires expected_revision, status and reason; reviewed needs
source-review evidence.
Setting validated is unavailable until a trusted experimental evidence verifier exists.

## HTTP

| Endpoint | Purpose |
| --- | --- |
| GET /dashboard.html | HTML prototype |
| GET /api/rsi/snapshot | Selected demo run and events |
| GET /api/rsi/taxonomy?backend=ascend | Seven-layer map |
| GET /api/rsi/knowledge | Requires scope/backend/component_version/workload query fields |
| POST /api/rsi/commands | JSON action/expected_revision/idempotency_key/payload |

Host must match a loopback address and the actual port. A supplied Origin must match the local origin.
POST requires JSON, at most 64 KiB and a demo run; no shell is executed. Invalid/conflicting commands return 409;
origin/run boundary rejection returns 403. SSE, remote authentication, role authorization, artifact storage and
deployment adapters are not implemented.
