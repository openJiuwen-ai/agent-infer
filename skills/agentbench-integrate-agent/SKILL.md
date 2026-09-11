---
name: agentbench-integrate-agent
description: Use when adding or hardening a third-party coding-agent runtime in AgentBench; discovers upstream behavior, implements the shared adapter contract, and verifies lifecycle, identity, isolation, artifacts, and benchmark regression gates.
---

# agentbench-integrate-agent

Integrate one external coding-agent runtime through AgentBench's shared
`AgentRunRequest` / `AgentRunResult` boundary. Preserve the runtime's native
transport and lifecycle while proving the common benchmark contracts.

## WHEN TO INVOKE

- Add a runtime under `agentinfer/agentbench/agents/`.
- Harden an existing adapter's lifecycle, identity, isolation, artifacts, or
  regression coverage.
- Review an adapter against AgentBench extension requirements.

Do NOT invoke for changing only a prompt, running an existing benchmark, or
vendoring an upstream agent without an architecture decision.

## READ FIRST

Use the current implementation and these repository contracts as authority:

- [Benchmark subsystem](../../design/module/benchmarking/index.md)
- [BenchKit orchestration](../../design/module/benchmarking/benchkit-orchestration.md)
- [Agent runtime adapters](../../design/module/benchmarking/agent-runtime-adapters.md)
- [Request proxy and hints](../../design/module/benchmarking/request-proxy-and-hints.md)
- [Artifacts and evaluation](../../design/module/benchmarking/artifacts-and-evaluation.md)
- [Program identity](../../design/module/scheduling/program-identity.md)
- [Benchmark quickstart](../../docs/en/how-to/run-benchmark.md)

If documentation and code differ, verify both and report the mismatch.

## WORKFLOW

```text
DISCOVER -> DESIGN -> IMPLEMENT -> VERIFY -> SIMPLIFY -> PREPARE_REVIEW
```

### 1. DISCOVER

Inspect the repository, upstream source, CLI help, and a minimal local probe
before asking the user. Record:

- supported version and installation;
- headless command and prompt transport;
- model/base URL/key injection and initialization order;
- every mutable HOME/config/data/session/cache/log/temp path;
- process/service topology, readiness, completion, interruption, and cleanup;
- request schema, usage, and root/child identity source;
- native sandbox and bypasses;
- transcripts, logs, artifacts, and secret exposure risks.

Prove the non-interactive command, prompt delivery, completion signal, request
path, and cleanup with a mock endpoint or minimal real run. Return `BLOCKED`
when essential behavior is inferred rather than observed.

### 2. DESIGN

Implement the existing `AgentRuntime` contract:

- `agent_type`
- `required_endpoint`
- `usage_observer_class`
- `get_profile()`
- `preflight()`
- `run(AgentRunRequest) -> AgentRunResult`

Keep BenchKit unaware of runtime internals. Runtime-specific prompt transport,
processes, completion detection, identity bridge, transcripts, and isolation
remain inside the adapter. Define each profile's enforceable tool/permission
policy, topology, non-interactive behavior, completion invariant, and supported
identity level. Prompt wording alone is not policy enforcement.

Before coding, summarize the lifecycle, process ownership, profile matrix,
identity mapping, isolation claim, artifacts, and test matrix.

### 3. IMPLEMENT

Add only files with a clear responsibility under
`agentinfer/agentbench/agents/<agent>/`. Update every applicable registration
surface:

- `agents/registry.py`;
- `AgentConfig.type`, profile validation, and required endpoint;
- usage observer selection;
- config, registry, dispatch, and adapter tests;
- sample YAML/package data and user documentation.

Implement the smallest vertical slice in this order:

1. runtime/profile contract and registry entry;
2. config validation and executable capability preflight;
3. task-local state and upstream initialization;
4. process startup/readiness and proxy-routed prompt delivery;
5. one task deadline, completion classification, cancellation, and cleanup;
6. standard result, shared patch export, and secret-safe artifacts;
7. adapter tests, shared hermetic benchmark smoke, then minimal real-CLI smoke.

Do not add provider-specific fields to shared contracts without separate review.

### 4. VERIFY SIX GATES

1. **Task-local state and concurrency:** two concurrent tasks must not share
   mutable paths, ports, services, workspaces, artifacts, or identities.
2. **Lifecycle:** the task deadline covers setup through bounded finalization;
   timeout and cancellation stop runtime work, child processes/services, and
   streams. Re-raise `CancelledError` after cleanup.
3. **Online Program identity:** state whether support is none, root/session, or
   full lineage. Scheduler claims require identity before scheduling. Prefer
   canonical `vllm_xargs.agentic_context`; do not add agent-specific conversion
   to the Request Proxy.
4. **Execution isolation:** define exactly what is blocked. Test workspace
   writes and claimed outside-workspace, cross-task, absolute, relative, and
   symlink restrictions on the target OS. Fail closed if required isolation is
   unavailable.
5. **Artifacts and secrets:** use shared `export_patch(workspace)`; preserve
   runtime logs/transcript/error/result/request evidence without credentials.
   Keep completion, patch presence, and SWE-bench correctness distinct.
6. **Benchmark regression:** run `tests/agentbench/test_benchmark_smoke.py`.
   Keep its config/dataset/workspace/dispatch/proxy-server/fake-backend/trace/
   patch/finalizer path. Fakes remain test-only; never register a production
   fake runtime.

### 5. VERIFY BY EVIDENCE STRENGTH

Run affected layers in order:

1. function tests for profiles, settings, parsing, completion, and redaction;
2. module tests for runner lifecycle, failure, timeout, cancellation, cleanup,
   request path, and artifacts;
3. hermetic single-task and concurrent benchmark smoke;
4. real CLI smoke in the declared runtime environment;
5. real target-OS isolation and concurrent E2E;
6. external evaluator and repeated performance runs only for those claims.

Record the code scope, runtime/dependency versions, profile, host/OS,
model/backend/endpoint, exact command, result, and raw artifact path. Do not
reuse evidence from another version, profile, OS, endpoint, or isolation mode.

### 6. SIMPLIFY AND PREPARE REVIEW

Remove provider-private fallbacks, proxy conversion, copied sandbox logic,
text heuristics when native evidence exists, dead state, duplicated tests,
empty modules, and runtime knobs leaked into shared contracts. Keep one tested
path.

Prepare a reviewer guide covering capabilities, profiles, lifecycle, cleanup,
identity/isolation claims, artifacts/secrets, validation evidence, limitations,
and files grouped by responsibility.

## BLOCKED CONDITIONS

Stop or reduce the claim when the runtime lacks a reliable headless path,
prompt/completion evidence, proxy routing, task-local state, bounded cleanup,
the claimed identity/isolation level, secret-safe configuration, a supported
version probe, or required real environment evidence.

## DON'T

- Don't assume another adapter's transport, lifecycle, or sandbox applies.
- Don't infer online lineage from post-run transcripts.
- Don't treat a patch as correctness or one run as performance evidence.
- Don't install system dependencies, start remote services, run costly
  evaluators, commit, push, or publish artifacts without authorization.

## AFTER

Report passed evidence, open gates, limitations, files, and the next smallest
action. Call the integration complete only for claims backed by matching
evidence.
