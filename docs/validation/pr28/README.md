# PR #28 validation evidence

This directory addresses the [validation review](https://github.com/openJiuwen-ai/agent-infer/pull/28#pullrequestreview-5359696336)
and [bounded-download review](https://github.com/openJiuwen-ai/agent-infer/pull/28#discussion_r4139315795).
All linked artifacts are committed with the report; no access to the author's machine is needed to inspect them.
Paths inside historical raw files are preserved verbatim. Use the relative links below to open those files.

## Revision and environment

- Review base: `ff6b99c103a365d93979eb6e6777cf3dc330f85a`.
- Current checks: base plus the streaming-download changes in this revision. Exact tested source hashes,
  commands, environment and run status are recorded in [validation.json](validation.json).
- Validation host on 2026-10-08: Linux x86_64, 2 NVIDIA L20 GPUs (46068 MiB each), driver 580.178.04;
  Python 3.12.13, pytest 9.1.1, httpx 0.28.1, huggingface-hub 1.32.0, pydantic 2.13.5.
- No vLLM service was listening on port 8020, and the validation Python environment had no vLLM package.
  These current host observations do not establish the environment of the historical run.

## Current validation

| Check | Result | Raw output |
| --- | --- | --- |
| Focused Replay regression | 124 passed, 1 live test skipped | [pytest.txt](pytest.txt) |
| Complete Replay tests plus runner lifecycle, with the test-ID workaround below | 231 passed, 1 live test skipped | [pytest-replay-suite-no-id-escaping.txt](pytest-replay-suite-no-id-escaping.txt) |
| Real Inferact cold-cache download, 1 record | 212992 bytes received; 160912-byte subset | [download-smoke-1.txt](download-smoke-1.txt) |
| Real Inferact cold-cache download, 8 records | 1840084 bytes received; 1811711-byte subset | [download-smoke-8.txt](download-smoke-8.txt) |

The pinned upstream blob is 218648292 bytes. Both network checks used real HTTPS responses, bypassed existing
Hub blobs, wrote into temporary directories, and verified that a repeated subset lookup made no network request.
Received byte counts can vary with buffering. These checks establish bounded source preparation, not TTFT or
inference throughput improvements. Their exact commands are in the manifest.

The complete test command initially crashed inside pytest's parameter-ID escaping during collection
([traceback](pytest-replay-suite.txt), exit 139). Re-running with
`-o disable_test_id_escaping_and_forfeit_all_rights_to_community_support=True` allowed all tests to execute.
This option changes test names, not test selection or assertions. The normal focused command passed without it.

From the repository root, using a Python environment with the project dependencies installed:

```bash
python -m pytest -q \
  -o disable_test_id_escaping_and_forfeit_all_rights_to_community_support=True \
  tests/agentbench/replay tests/agentbench/test_runner_lifecycle.py
HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONPATH=. \
  python docs/validation/pr28/download_smoke.py --task-num 1
HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONPATH=. \
  python docs/validation/pr28/download_smoke.py --task-num 8
python docs/validation/pr28/verify.py
```

## Historical real-vLLM short trace

The preserved run started at `2026-09-28T11:30:04.755555+00:00` and completed successfully. It uses
8 synthetic sessions with 2 requests each, at maximum concurrency 4. It is not a full real TraceLab workload.

| Metric | Recomputed / recorded value |
| --- | --- |
| Completed / failed tasks | 8 / 0 |
| Successful / failed measured requests | 16 / 0 |
| Measured input / output tokens | 1152 / 114 |
| Source zero-output requests sent with one output token | 1 |
| Warmup input / output tokens, excluded from measured totals | 76 / 1 |

Artifacts:

- [Input trace](historical-live/live-eight-sessions.jsonl), [resolved config and lifecycle](historical-live/live-run/manifest.json).
- [Raw measured requests](historical-live/live-run/requests.jsonl), [parsed summary](historical-live/live-run/summary.json).
- [Execution summary](historical-live/live-run/replay-execution.json), [plan](historical-live/live-run/replay-plan.json).
- [Conversion manifest](historical-live/live-run/convert_result/manifest.json), [converted requests](historical-live/live-run/convert_result/requests.jsonl).
- [Warmup](historical-live/live-run/evidence/tracelab_warmup.json), [starting metrics](historical-live/live-run/evidence/vllm_metrics_start.prom),
  [ending metrics](historical-live/live-run/evidence/vllm_metrics_end.prom).

The recorded model is `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`; task/run timeouts are disabled,
`sample_seed=0`, `interval_mode=trace`, and the inter-round gap scale is zero.
The original PR reported this live-test command:

```bash
REPLAY_LIVE_BASE_URL=http://127.0.0.1:8020 .venv/bin/pytest -q \
  tests/agentbench/replay/test_tracelab.py::test_tracelab_live_eight_tasks_four_concurrent
```

The historical exact commit, vLLM version, GPU inventory, server launch command and console log were not recorded
in the available artifacts. They are marked unknown in the manifest, rather than inferred from today's host.
In particular, the historical warmup was 76 tokens; current round-zero-only selection would use 33 tokens for
this input. Therefore this run must not be presented as live validation of the current PR head.

## Live reproduction and full-run status

To reproduce the short-trace correctness check against a compatible vLLM service serving the model above:

```bash
REPLAY_LIVE_BASE_URL=http://127.0.0.1:8020 python -m pytest -q \
  tests/agentbench/replay/test_tracelab.py::test_tracelab_live_eight_tasks_four_concurrent \
  --basetemp=results/pr28-live-8x4
```

Use a new output directory for each run. Record the tested commit, dirty state, GPU inventory, vLLM version,
model revision and exact service launch command alongside the new artifacts. The original launch command
cannot be reconstructed reliably from the saved metrics.

Current-head live validation is **NOT RUN** because no usable local vLLM service is available.
Full real-trace 8-task/4-concurrency replay is also **NOT RUN**. The original PR reported a maximum context
requirement of 420339 tokens against a 100000-token backend limit; this report does not claim to remeasure it.
A context-compatible subset is a separate validation workload and must be identified by its source hash and
selection criteria, rather than described as completion of that original full trace.

[verify.py](verify.py) checks artifact SHA256 values and independently recomputes the historical request/token
totals. It does not turn historical evidence or skipped tests into current-head live validation.
