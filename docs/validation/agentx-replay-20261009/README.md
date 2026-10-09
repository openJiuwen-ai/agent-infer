# AgentX replay validation: dual L20, 2026-10-08/09

This evidence covers eight complete selected sessions with four concurrent runtime
sessions. Each complete live run contains 254 requests with 13,957,952 input tokens
and 162,310 output tokens. The two baseline cold starts were tested on `e6db2ee`;
the review-fix run tests the accompanying code changes, identified by
`tested-code.sha256`. The server was restarted between runs. Requests were not
sent to the model before the review-fix run; readiness checks used `/health`.

## Configuration and scope

See [validation-manifest.json](validation-manifest.json) for exact host, GPU,
software versions, commands, source session IDs, archive hashes, and parsed
results. [replay.yaml](replay.yaml) is the exact YAML used by the three live runs.
Paths record the original test host; change paths and use a fresh result directory
when reproducing elsewhere. Model weights and the source dataset are not bundled.

The source is the public
[AgentX dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k).
[select_sessions.py](select_sessions.py) selects the first eight complete sessions
where every output is positive, each input plus output is at most 100,000 tokens,
and the source session span is at most 3,600 seconds. Selection does not truncate
sessions. The generated subset's SHA256 and selected IDs are in the manifest.

The full dataset was **not** replayed live. These live runs use
`max_inflight_requests=1` and `trace_same_agent_gap_scale=0`, so they do not
validate the default per-session concurrency of eight or the original trace gaps.
The earlier PR's 21-request experiment is not included in these archives; these
dated runs provide independently inspectable live evidence.
The complete 393-session, 68,266-request dataset was converted, validated, and
analyzed offline after the review fix; see [full-validation.json](full-validation.json).

## Raw artifacts and verification

Each `*.tar.gz` contains the raw client/server logs, the result manifest,
`requests.jsonl`, source analysis, plan, execution records, parsed `summary.json`,
and the YAML. Extract them to inspect individual source requests and outcomes.
The archives are tracked with this PR so reviewers can download them directly.

```bash
sha256sum -c SHA256SUMS
tar -xzf baseline-cold1.tar.gz
tar -xzf baseline-cold2.tar.gz
tar -xzf review-fix.tar.gz
python compare_runs.py baseline-cold1 review-fix
python compare_runs.py baseline-cold2 review-fix
```

[comparison.json](comparison.json) compares the review-fix run with the first
baseline by source session/request identity, checking status, actual input/output
token counts, planned targets, source digest, and workload fingerprint.
[comparison-cold2.json](comparison-cold2.json) performs the same checks against
the second baseline. Both comparisons report zero mismatches. All three runs
completed eight tasks and 254 requests, with 13,957,952 input tokens and 162,310
output tokens. The review-fix TTFT p50/p99 were 0.570/16.572 seconds.
[pytest.log](pytest.log) records the complete AgentBench unit-test result;
[pre-commit.log](pre-commit.log) records the full-repository static checks.

## Runtime setup

The server used Python 3.11.15, vLLM 0.19.0, PyTorch 2.10.0+cu128,
FastAPI 0.115.12, and Starlette 0.46.2. The model was
Qwen3-Coder-30B-A3B-Instruct-FP8, tensor parallel size two, with a 100,000-token
context window. GPU workers were stopped and GPU memory released between cold
starts. This host's cached Python 3.12.13 interpreter failed on `re.escape("")`;
the review tests and live replay client also used an isolated Python 3.11 environment.

The palette and token rendering algorithm are unchanged by the review fix.
Completed AgentX requests release their input snapshots rather than retaining
them as continuation context. Rendering remains inside the session semaphore;
recorded scheduler lag includes semaphore wait and rendering time.
