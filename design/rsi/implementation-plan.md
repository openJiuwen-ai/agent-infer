# RSI dense-feedback bootstrap implementation plan

## Goal and scope

Add an opt-in, CPU-only RSI control scaffold with local dense-feedback evaluation,
scoped knowledge, a demonstration dashboard, architecture figures, and a Chinese
presentation. Keep the existing serving path independent of RSI. Model weights stay
fixed. Real vLLM/Ascend profiling, hardware experiments, agent execution, and service
deployment are integration work, not claimed by this bootstrap.

## Implementation

1. Define a shared logical backend taxonomy and typed feedback records. Evaluate
   frozen checks against one exact candidate/baseline/backend/version/workload scope.
   Preserve missing, profiled, and unavailable evidence as inconclusive.
2. Implement SQLite-backed demonstration state transitions, command idempotency,
   revision checks, budget checks, and scoped knowledge records. Knowledge proposals
   cannot mark themselves empirically validated.
3. Add a standard-library CLI and a loopback demonstration HTTP server. Ship the
   standalone HTML as package data. Neither simulated commands nor dashboard actions
   call production deployment APIs.
4. Extend the dashboard with correctness, system-behavior, and performance feedback,
   local refinement paths, evidence details, and per-backend engine layers.
5. Document architecture, scope, public interfaces, and usage in Chinese and English.
   Include numbered figures, editable diagram sources, and a narrated PPTX.
6. Run focused CPU tests, syntax/format/package checks, and review the presentation
   and dashboard. Do not start GPU/NPU services or claim real performance results.
7. Open a design issue and draft PR, following the repository template. All commits
   carry DCO sign-off. Include exact validation and not-run statements.

## Review criteria

- Importing AgentInfer does not import RSI, vLLM, or a model runtime indirectly.
- Missing evidence, scope mismatch, or profiled timings cannot produce a performance pass.
- Human commands cannot bypass phase, budget, or idempotency checks.
- Knowledge retrieval preserves backend/version/workload applicability and counterexamples.
- All synthetic fixtures, UI values, and simulated transitions are explicitly labelled.
- Engine layers describe logical ownership; parallelism and operations can span layers.
- The PR contains no local databases, credentials, private paths, or generated test logs.
