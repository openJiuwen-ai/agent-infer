# M2 Plugin Template — Adversarial Review vs real vLLM 0.21.0

**Date:** 2026-06-03
**Method:** 3 independent agents read the actual vLLM 0.21.0 source
(`vllm/utils/import_utils.py`, `vllm/config/scheduler.py`,
`vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/sched/request_queue.py`,
`vllm/v1/request.py`) and tried to refute that `targets/scheduling/plugin_template.py`
would load and run.

## Findings and resolutions

| # | Severity | Finding | Fix applied |
|---|----------|---------|-------------|
| 1 | **BLOCKER** | `--scheduler-cls generated_scheduler:EvolvedScheduler` (colon) crashes: vLLM resolves the flag via `resolve_obj_by_qualname` → `qualname.rsplit('.', 1)`, which splits on a DOT only. Colon → `ValueError: not enough values to unpack` (cf. vLLM #17188). | `SCHEDULER_QUALNAME = "generated_scheduler.EvolvedScheduler"` (dot). Fixed in template docstrings + `backend.DEFAULT_SCHEDULER_CLS`. Tested. |
| 2 | MAJOR | Module must be importable as the bare name `generated_scheduler` → exactly `<plugin_dir>/generated_scheduler.py`, top-level, with `PYTHONPATH=<plugin_dir>`. | Added `render_and_write(plugin_dir, src)` writing exactly `generated_scheduler.py`. `build_docker_command` mounts `/plugins` + `PYTHONPATH=/plugins`. Tested. |
| 3 | MAJOR | `self.waiting` is a `RequestQueue`: `FCFSRequestQueue(deque)` (reorderable) or `PriorityRequestQueue` (no-arg init, no `clear`/`append`, heap re-imposes order). Old `type(self.waiting)(reordered)` + `clear()/append()` fallback both raise on priority and were silently swallowed → no-op. | Branch on `isinstance(self.waiting, deque)`: reorder FCFS in place; **skip priority cleanly** (documented no-op). Tested with fake deque + fake priority queue. |
| 4 | MAJOR | Inheriting the 8-arg `Scheduler.__init__` is compatible only under the exact 0.21.0 pin. | Pin already enforced; CI signature gate deferred to AC-1 (M0/GPU). Documented. |
| 5 | MINOR | V1 `Request` has **no `num_cached_tokens`** → `prefix_cached_tokens` was always 0. | Use `num_computed_tokens` (folds in prefix-cache hits for waiting reqs) as the proxy. |
| 6 | **BLOCKER (corrected 2026-07-27)** | Real `Request.num_output_tokens` is the number already generated and is zero for a newly waiting request. Treating it as planned decode length silently disables decode-aware policies such as D3Q. vLLM 0.21.0 exposes the requested budget as `Request.max_tokens`. | Read `request.max_tokens` first, then `sampling_params.max_tokens`; use current generated length only as an old-version fallback. A unit test pins the mapping. |
| 7 | MINOR | `arrival_time` is wall-clock `time.time()`, template used `time.monotonic()`. | Use `time.time()` to match. |
| 8 | MINOR | "invoked" marker prints before reorder → green AC-1 ≠ policy effective. | Added `PLUGIN_REORDERED_MARKER` (printed only after an actual reorder) + `assert_plugin_effective()`. |

## Confirmed correct (no change)

- Subclassing `Scheduler` + overriding `schedule(self) -> SchedulerOutput` via
  `super().schedule()` is the right V1 API.
- No `vllm.general_plugins` entry-point registration is required.
- Bare `print(..., flush=True)` is captured into container logs (prefixed
  `(EngineCore pid=...)`); substring grep (not line-anchored) matches — confirmed.

## Residual GPU-only validation (AC-1, when hardware is available)

- Under **FCFS** the reorder sticks within `self.waiting`, but 0.21.0 drains a
  second per-step `self.skipped_waiting` queue first, so the policy **biases**
  admission rather than strictly controlling it. Acceptable; documented.
- Verify the constructor signature + `schedule()->SchedulerOutput` on the pinned
  image; assert `assert_plugin_effective` (not just invoked) on a real run.

Sources: github.com/vllm-project/vllm @ v0.21.0 (files above), issue #17188.
