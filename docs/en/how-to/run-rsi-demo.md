# Run the offline RSI scaffold and Dashboard

Use Python 3.10+ at the repository root. These commands require only the standard library and do not load a model,
start vLLM or allocate a GPU/NPU. Hardware and deployment validation is outside this delivery.

```bash
python -m agentinfer.rsi taxonomy --backend cuda
python -m agentinfer.rsi taxonomy --backend ascend
python -m agentinfer.rsi evaluate examples/rsi/feedback.json --output .rsi-demo/report.json
python -m agentinfer.rsi demo --run-dir .rsi-demo
python -m agentinfer.rsi status --run-dir .rsi-demo
python -m agentinfer.rsi serve --run-dir .rsi-demo --port 8877
```

Open <http://127.0.0.1:8877/dashboard.html>, or directly open `agentinfer/rsi/dashboard/static/dashboard.html`
for offline interaction. A file URL cannot access the local API snapshot.
The sample evaluation must be INCONCLUSIVE: missing probes and synthetic data cannot establish acceptance.
`demo` initializes stage 7; running it again preserves existing state.

## Dashboard walkthrough

1. Overview: inspect stage, candidate/accepted/active/last_good, budget and task outcomes.
2. Evolution loop: follow steps 1–10 and the failure, missing-evidence, recovery and knowledge feedback paths.
3. Changes: inspect illustrative file diffs, component attribution and linked evidence.
4. Dense feedback: filter by backend and layer; connect hypotheses, checks, evidence gaps and next discriminating tests.
5. Knowledge: inspect system/routing/harness/backend-layer knowledge, scope and lifecycle.
6. Intervention: simulate pause, budgets, hypotheses, exclusion, activation and recovery; inspect the audit trail.

HTML controls operate on browser-local simulation in localStorage. CLI/HTTP operate on separate SQLite demo state;
they are not automatically synchronized.
The API snapshot feature only reads server state. Acceptance, activation, observation and rollback commands use
explicit `demo_` names.
The server binds IPv4 loopback only. It is a trusted-local demonstration interface, without production authentication
or deployment support.

## CLI intervention

Read revision from `status` before sending a command. Revision 6 below applies only to a fresh, unchanged demo.
The identical request/key returns its first response; reusing the key with changed content or using an outdated
revision is rejected.

```bash
python -m agentinfer.rsi command --run-dir .rsi-demo --action pause --expected-revision 6 --idempotency-key pause-demo-1
python -m agentinfer.rsi status --run-dir .rsi-demo
python -m agentinfer.rsi knowledge --scope demo --backend cuda --component-version demo-unverified --workload demo
```

Use backend `ascend` for that backend or `agnostic` for shared contracts. All seeds are unvalidated drafts, not
experimental conclusions.

## CPU verification and later integration

```bash
python -m unittest discover -s tests/rsi -p "test_*.py" -v
```

This checks CPU state and data contracts only. Later integration needs per-backend protocol smoke checks, reference
numerical tests,
cancellation/recovery checks, isolated paired performance trials, real agent tasks, contention and deployment recovery.
None of these hardware results is claimed here.

See [architecture](../explanation/rsi-architecture.md), [API reference](../reference/rsi-api.md) and
the [implementation plan](../../../design/rsi/implementation-plan.md) and [future component
map](../../../design/rsi/component-plan.md).
