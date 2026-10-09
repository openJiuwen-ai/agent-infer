"""Render an evolved scheduling policy as a real vLLM V1 scheduler plugin.

Grounded in the T0.4 API study **and** an adversarial review against the actual
vLLM 0.21.0 source (see docs/reviews/m2_plugin_review_2026-06-03.md):

* vLLM 0.21 enables asynchronous scheduling by default on the target executor,
  so a same-caliber custom scheduler subclasses
  ``vllm.v1.core.sched.async_scheduler.AsyncScheduler`` and overrides
  ``schedule(self) -> SchedulerOutput``. No ``__init__`` override and no
  entry-point registration.
* It is loaded with ``--scheduler-cls generated_scheduler.EvolvedScheduler``.
  vLLM resolves this via ``rsplit('.', 1)`` — it MUST be a dotted qualname
  (a colon ``mod:Class`` crashes at startup), and the module must be importable
  by the bare name ``generated_scheduler`` (i.e. saved as
  ``<plugin_dir>/generated_scheduler.py`` with ``PYTHONPATH=<plugin_dir>``).
* ``self.waiting`` is a ``RequestQueue``: ``FCFSRequestQueue`` (a ``deque``
  subclass, reorderable in place) or ``PriorityRequestQueue`` (a heap with no
  ``clear``/``append`` that re-imposes (priority, arrival) order). The bridge
  reorders the deque-backed FCFS queue and **cleanly skips** the priority queue
  (where external reordering is meaningless) instead of relying on a swallowed
  exception.

``render_plugin(policy_source)`` injects the evolved ``schedule_batch`` source
into the template. ``render_and_write`` writes it to the exact filename vLLM
needs. The bridge logic is unit-tested against fake FCFS (deque) and priority
queues without a GPU.
"""
from __future__ import annotations

from pathlib import Path

_POLICY_MARKER = "# >>> EVOLVED POLICY INSERTED HERE <<<"

# Loading contract (dot-form qualname; module importable as `generated_scheduler`).
PLUGIN_MODULE_NAME = "generated_scheduler"
PLUGIN_FILENAME = "generated_scheduler.py"
SCHEDULER_QUALNAME = "generated_scheduler.EvolvedScheduler"

# Log markers: "invoked" proves the plugin loaded+ran; "reordered" proves the
# evolved policy actually changed admission order (a green "invoked" alone does
# NOT prove the policy had any effect — e.g. under priority scheduling).
PLUGIN_INVOKED_MARKER = "vllm-evolve: scheduler plugin invoked"
PLUGIN_REORDERED_MARKER = "vllm-evolve: waiting reordered"
# "fallback" proves the policy raised and we reverted to stock ordering -> the
# candidate is NOT valid for that run (M-B2: effective gate rejects any fallback).
PLUGIN_FALLBACK_MARKER = "vllm-evolve: policy fallback"
PLUGIN_ACTIVE_MARKER = "vllm-evolve: policy mechanism active"
PLUGIN_INACTIVE_MARKER = "vllm-evolve: policy mechanism inactive"
PLUGIN_PREEMPTED_MARKER = "vllm-evolve: running request preempted"
PLUGIN_POLICY_CALL_MARKER = "vllm-evolve: policy decision evaluated"
PLUGIN_DEFERRED_MARKER = "vllm-evolve: waiting request deferred"
PLUGIN_FORCED_ADMIT_MARKER = "vllm-evolve: starvation bound forced admission"


def reorder_waiting(waiting_ids: list[str], priority_ids: list[str]) -> list[str]:
    """Stable reorder: requested ``priority_ids`` (that exist in waiting) first,
    in the given order; all other waiting ids keep their original relative order.
    """
    present = set(waiting_ids)
    pri = [rid for rid in priority_ids if rid in present]
    seen = set(pri)
    rest = [rid for rid in waiting_ids if rid not in seen]
    return pri + rest


_TEMPLATE = '''\
"""Auto-generated vLLM V1 scheduler plugin (vllm-evolve).

Load with (DOT qualname, module on PYTHONPATH):
    vllm serve <model> --scheduler-cls generated_scheduler.EvolvedScheduler
"""
from __future__ import annotations

import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import islice

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

_PLUGIN_INVOKED_MARKER = "vllm-evolve: scheduler plugin invoked"
_PLUGIN_REORDERED_MARKER = "vllm-evolve: waiting reordered"
# Emitted when the policy raises and we fall back to stock ordering. NON-silent on
# purpose (M-B2): a candidate that loaded but then errored + fell back is NOT a
# valid candidate — the effective gate rejects any run with a fallback.
_PLUGIN_FALLBACK_MARKER = "vllm-evolve: policy fallback"
_PLUGIN_ACTIVE_MARKER = "vllm-evolve: policy mechanism active"
_PLUGIN_INACTIVE_MARKER = "vllm-evolve: policy mechanism inactive"
_PLUGIN_PREEMPTED_MARKER = "vllm-evolve: running request preempted"
_PLUGIN_POLICY_CALL_MARKER = "vllm-evolve: policy decision evaluated"
_PLUGIN_DEFERRED_MARKER = "vllm-evolve: waiting request deferred"
_PLUGIN_FORCED_ADMIT_MARKER = "vllm-evolve: starvation bound forced admission"
# Per-run nonce injected by the harness (native.serve_vllm). Markers are emitted as
# "<marker> <nonce>" so a candidate that prints the static marker text cannot satisfy
# the gate — it can't know this per-run secret (Codex HIGH#3).
_VE_NONCE = os.environ.get("VE_MARKER_NONCE", "")


def _emit(marker: str) -> None:
    print(f"{marker} {_VE_NONCE}".rstrip(), flush=True)


def _ve_raw_decode_budget(request) -> int:
    budget = getattr(request, "max_tokens", None)
    if budget is None:
        sampling_params = getattr(request, "sampling_params", None)
        budget = getattr(sampling_params, "max_tokens", 0)
    return int(budget or 0)


def _ve_raw_prompt_budget(request) -> int:
    budget = getattr(request, "num_prompt_tokens", None)
    if budget is None:
        budget = len(getattr(request, "prompt_token_ids", None) or [])
    return int(budget or 0)


@dataclass
class RequestInfo:
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    arrival_time_s: float
    is_prefill: bool
    kv_blocks_used: int
    prefix_cached_tokens: int
    num_preemptions: int = 0
    has_prefix_hint: bool = False
    session_id: object | None = None
    generated_output_tokens: int = 0
    waiting_age_s: float = 0.0

    @property
    def remaining_prompt_tokens(self) -> int:
        return self.num_prompt_tokens - self.num_computed_tokens

    @property
    def remaining_output_tokens(self) -> int:
        return max(0, self.num_output_tokens - self.generated_output_tokens)


@dataclass
class ScheduleDecision:
    prefill_batch: list
    decode_batch: list
    preempt_ids: list = field(default_factory=list)
    defer_ids: list = field(default_factory=list)
    mechanism_applicable: bool | None = None


def _reorder_waiting(waiting_ids, priority_ids):
    present = set(waiting_ids)
    pri = [rid for rid in priority_ids if rid in present]
    seen = set(pri)
    rest = [rid for rid in waiting_ids if rid not in seen]
    return pri + rest


def _prefix_key(request):
    hashes = getattr(request, "block_hashes", None) or []
    if hashes:
        first = hashes[0]
        value = getattr(first, "block_hash", None)
        if value is None:
            value = getattr(first, "hash_value", None)
        return value if value is not None else repr(first)
    # vLLM may not materialize block_hashes until after admission. Waiting
    # requests still expose exact prompt token ids, so use one physical block
    # as a truthful pre-admission cohort key. This mirrors Frontier's explicit
    # prefix hash instead of silently reporting "no prefix signal".
    token_ids = getattr(request, "prompt_token_ids", None) or []
    return tuple(token_ids[:16]) if len(token_ids) >= 16 else None


def _request_to_info(
    request, now_s, is_prefill, has_prefix_hint=False, session_id=None,
):
    # All field names verified against vLLM 0.21.0 vllm/v1/request.py.
    num_prompt = getattr(request, "num_prompt_tokens", None)
    if num_prompt is None:
        num_prompt = len(getattr(request, "prompt_token_ids", []) or [])
    num_prompt = int(num_prompt)
    num_computed = int(getattr(request, "num_computed_tokens", 0) or 0)
    # ``Request.num_output_tokens`` is the number ALREADY generated, so it is
    # zero while a new request waits.  The scheduler-visible planned decode
    # length is ``Request.max_tokens`` (copied from SamplingParams.max_tokens).
    # This distinction is essential for decode-aware evolved policies.
    num_output = getattr(request, "max_tokens", None)
    if num_output is None:
        sampling_params = getattr(request, "sampling_params", None)
        num_output = getattr(sampling_params, "max_tokens", None)
    if num_output is None:
        # Compatibility fallback for older request objects.  This is current
        # generated length, not the preferred requested maximum.
        num_output = getattr(request, "num_output_tokens", 0) or 0
    arrival = getattr(request, "arrival_time", now_s)  # wall-clock (time.time)
    output_token_ids = getattr(request, "output_token_ids", None) or []
    return RequestInfo(
        request_id=str(getattr(request, "request_id", "")),
        num_prompt_tokens=num_prompt,
        num_computed_tokens=num_computed,
        num_output_tokens=int(num_output),
        arrival_time_s=float(arrival),
        is_prefill=is_prefill,
        kv_blocks_used=0,
        # V1 Request has no num_cached_tokens; num_computed_tokens folds in
        # prefix-cache hits for waiting requests, so it is the best proxy.
        prefix_cached_tokens=num_computed if is_prefill else 0,
        num_preemptions=int(getattr(request, "num_preemptions", 0) or 0),
        has_prefix_hint=bool(has_prefix_hint),
        session_id=session_id,
        generated_output_tokens=len(output_token_ids),
        waiting_age_s=max(0.0, now_s - float(arrival)),
    )


# >>> EVOLVED POLICY INSERTED HERE <<<


class EvolvedScheduler(AsyncScheduler):
    """vLLM V1 scheduler that applies the evolved policy as a priority over
    ``self.waiting`` (deque-backed FCFS queue) and then delegates allocation to
    the stock scheduler. Under priority scheduling the waiting queue is a heap
    that re-imposes its own order, so the reorder is skipped (documented no-op)."""

    _ve_logged = False
    _ve_reordered = False
    _ve_fallback = False
    _ve_last_policy_fingerprint = None
    _ve_active_logged = False
    _ve_inactive_logged = False
    _ve_allow_active_preemption = bool(
        globals().get("ALLOW_ACTIVE_PREEMPTION", False)
    )
    _ve_active_preemption_only = bool(
        globals().get("ACTIVE_PREEMPTION_ONLY", False)
    )
    _ve_active_preemption_min_output_tokens = int(
        globals().get("ACTIVE_PREEMPTION_MIN_OUTPUT_TOKENS", 0) or 0
    )
    _ve_active_preemption_requires_first_token = bool(
        globals().get("ACTIVE_PREEMPTION_REQUIRES_FIRST_TOKEN", False)
    )
    # Optional bounded observation surface for natural-release controllers.
    # Zero preserves the historical whole-queue bridge exactly. Active
    # preemption keeps the complete waiting view because _preempt_request may
    # prepend a victim after the policy input has been materialized.
    _ve_waiting_window = max(
        0, int(globals().get("POLICY_WAITING_WINDOW", 0) or 0)
    )
    _ve_policy_needs_running_requests = bool(
        globals().get("POLICY_NEEDS_RUNNING_REQUESTS", True)
    )
    _ve_global_recoverable_index = bool(
        globals().get("POLICY_GLOBAL_RECOVERABLE_INDEX", False)
    )
    _ve_index_width = max(
        1, int(globals().get("POLICY_INDEX_WIDTH", 4) or 4)
    )
    _ve_index_slo_s = max(
        0.001, float(globals().get("POLICY_TTFT_SLO_S", 120.0) or 120.0)
    )
    _ve_max_defer_events = max(
        1, int(globals().get("MAX_DEFER_EVENTS", 8) or 8)
    )
    _ve_defer_counts = {}

    def _ve_index_arrival(self, request):
        request_id = str(getattr(request, "request_id", ""))
        if not request_id or int(
            getattr(request, "num_computed_tokens", 0) or 0
        ) > 0:
            return
        active = getattr(self, "_ve_index_active", None)
        if active is None:
            active = self._ve_index_active = {}
            self._ve_prompt_heap = []
            self._ve_residency_heap = []
        prompt = _ve_raw_prompt_budget(request)
        output = _ve_raw_decode_budget(request)
        arrival = float(getattr(request, "arrival_time", time.time()))
        active[request_id] = request
        # Both heaps use the same deterministic tail so request objects are
        # never compared when costs tie.
        tail = (prompt, arrival, request_id, request)
        heappush(self._ve_prompt_heap, (prompt, *tail))
        heappush(self._ve_residency_heap, (prompt + output, *tail))

    def _ve_live_index_frontier(self, heap, now_s):
        active = getattr(self, "_ve_index_active", None)
        if not active:
            return []
        kept = []
        candidates = []
        while heap and len(candidates) < type(self)._ve_index_width:
            entry = heappop(heap)
            request_id = entry[-2]
            request = entry[-1]
            if active.get(request_id) is not request:
                continue
            # Cancellation and non-indexed lifecycle transitions must not
            # leave a stale raw object eligible for a deep-queue move. Inspect
            # the request's O(1) lifecycle state; never search the deep deque.
            status = getattr(request, "status", None)
            status_name = str(getattr(status, "name", status or ""))
            if status is not None and "WAITING" not in status_name:
                active.pop(request_id, None)
                continue
            arrival = float(getattr(request, "arrival_time", now_s))
            if now_s - arrival >= type(self)._ve_index_slo_s:
                # Expiry removes only index eligibility. The raw request stays
                # in the real FCFS deque and remains completion-eligible.
                active.pop(request_id, None)
                continue
            kept.append(entry)
            candidates.append(request)
        for entry in kept:
            heappush(heap, entry)
        return candidates

    def add_request(self, request):
        # Decode-budget classification belongs on the cold arrival path, not
        # the per-token scheduling path. Short-only workloads therefore retain
        # an O(1) delegation to stock AsyncScheduler.
        if (
            type(self)._ve_active_preemption_only
            and _ve_raw_decode_budget(request)
            >= type(self)._ve_active_preemption_min_output_tokens
        ):
            self._ve_has_seen_elephant = True
        result = super().add_request(request)
        if type(self)._ve_global_recoverable_index:
            self._ve_index_arrival(request)
        return result

    def schedule(self):
        now = time.time()
        scheduled_timestamp = time.monotonic()
        preempted = []
        deferred = []
        policy_fingerprint = None
        if not type(self)._ve_logged:
            _emit(_PLUGIN_INVOKED_MARKER)
            type(self)._ve_logged = True
        try:
            allow_active_preemption = type(self)._ve_allow_active_preemption
            active_preemption_only = type(self)._ve_active_preemption_only
            if not self.waiting:
                if (
                    active_preemption_only
                    and not type(self)._ve_inactive_logged
                ):
                    _emit(_PLUGIN_INACTIVE_MARKER)
                    type(self)._ve_inactive_logged = True
                return super().schedule()
            sc = getattr(self, "scheduler_config", None)
            max_tokens = int(getattr(sc, "max_num_batched_tokens", 8192) or 8192)
            max_seqs = int(getattr(sc, "max_num_seqs", 256) or 256)
            # Admission policy cannot change anything while all sequence slots
            # are occupied. This check must precede *all* queue materialization:
            # copying a long waiting deque on every decode tick imposed a large
            # bridge tax even though no admission could change.
            if len(self.running) >= max_seqs and not allow_active_preemption:
                return super().schedule()
            if active_preemption_only:
                if not getattr(self, "_ve_has_seen_elephant", False):
                    if not type(self)._ve_inactive_logged:
                        _emit(_PLUGIN_INACTIVE_MARKER)
                        type(self)._ve_inactive_logged = True
                    return super().schedule()
                eligible_victim = any(
                    int(getattr(request, "num_preemptions", 0) or 0) == 0
                    and _ve_raw_decode_budget(request)
                    >= type(self)._ve_active_preemption_min_output_tokens
                    and (
                        not type(
                            self
                        )._ve_active_preemption_requires_first_token
                        or bool(getattr(request, "output_token_ids", None))
                    )
                    for request in self.running
                )
                if len(self.running) < max_seqs or not eligible_victim:
                    if not type(self)._ve_inactive_logged:
                        _emit(_PLUGIN_INACTIVE_MARKER)
                        type(self)._ve_inactive_logged = True
                    return super().schedule()
            indexed_view = (
                type(self)._ve_global_recoverable_index
                and not allow_active_preemption
                and isinstance(self.waiting, deque)
            )
            waiting_window = (
                type(self)._ve_waiting_window
                if not indexed_view
                and not allow_active_preemption
                and isinstance(self.waiting, deque)
                else 0
            )
            if indexed_view:
                indexed_raw = [
                    self.waiting[0],
                    *self._ve_live_index_frontier(
                        getattr(self, "_ve_residency_heap", []), now
                    ),
                    *self._ve_live_index_frontier(
                        getattr(self, "_ve_prompt_heap", []), now
                    ),
                ]
                seen_index_ids = set()
                waiting_list = []
                for request in indexed_raw:
                    request_id = str(getattr(request, "request_id", ""))
                    if request_id in seen_index_ids:
                        continue
                    seen_index_ids.add(request_id)
                    waiting_list.append(request)
            else:
                waiting_list = (
                    list(islice(self.waiting, waiting_window))
                    if waiting_window
                    else list(self.waiting)
                )
            running_list = (
                list(self.running)
                if allow_active_preemption
                or type(self)._ve_policy_needs_running_requests
                else []
            )
            waiting_fingerprint = tuple(
                str(getattr(r, "request_id", "")) for r in waiting_list
            )
            if allow_active_preemption:
                # Active policies may become applicable when a running request
                # emits its first token. Include exactly that lifecycle edge
                # and preemption count, while avoiding a full policy call on
                # every subsequent decode token.
                running_lifecycle = tuple(
                    (
                        str(getattr(r, "request_id", "")),
                        int(getattr(r, "num_preemptions", 0) or 0),
                        bool(getattr(r, "output_token_ids", None)),
                    )
                    for r in running_list
                )
                policy_fingerprint = (waiting_fingerprint, running_lifecycle)
            else:
                policy_fingerprint = waiting_fingerprint
            if policy_fingerprint == type(self)._ve_last_policy_fingerprint:
                return super().schedule()
            prefix_keys = [_prefix_key(r) for r in waiting_list]
            prefix_counts = Counter(key for key in prefix_keys if key is not None)
            waiting_infos = [
                _request_to_info(
                    r, now, True,
                    has_prefix_hint=key is not None and prefix_counts[key] > 1,
                    session_id=(
                        key
                        if key is not None and prefix_counts[key] > 1
                        else None
                    ),
                )
                for r, key in zip(waiting_list, prefix_keys)
            ]
            running_infos = [_request_to_info(r, now, False) for r in running_list]
            block_pool = getattr(
                getattr(self, "kv_cache_manager", None), "block_pool", None
            )
            available_kv_blocks = (
                int(block_pool.get_num_free_blocks())
                if block_pool is not None
                and hasattr(block_pool, "get_num_free_blocks")
                else 1 << 30
            )
            decision = schedule_batch(
                waiting_infos,
                running_infos,
                max_tokens,
                max_seqs,
                available_kv_blocks,
                0.0,
            )
            _emit(_PLUGIN_POLICY_CALL_MARKER)
            applicable = getattr(decision, "mechanism_applicable", None)
            if applicable is True and not type(self)._ve_active_logged:
                _emit(_PLUGIN_ACTIVE_MARKER)
                type(self)._ve_active_logged = True
            elif applicable is False and not type(self)._ve_inactive_logged:
                _emit(_PLUGIN_INACTIVE_MARKER)
                type(self)._ve_inactive_logged = True
            preempt_ids = {
                str(request_id)
                for request_id in (getattr(decision, "preempt_ids", None) or [])
            }
            if allow_active_preemption and preempt_ids:
                for request in list(self.running):
                    request_id = str(getattr(request, "request_id", ""))
                    if request_id not in preempt_ids:
                        continue
                    self.running.remove(request)
                    self._preempt_request(request, scheduled_timestamp)
                    if bool(getattr(sc, "async_scheduling", False)):
                        # Match vLLM's forced-preemption contract: the latest
                        # in-flight async token must not be emitted twice after
                        # the request resumes.
                        request.num_output_placeholders = 0
                        request.discard_latest_async_tokens = True
                    preempted.append(request_id)
                    _emit(_PLUGIN_PREEMPTED_MARKER)
                if preempted:
                    # vLLM's reset_prefix_cache(force=True) performs the same
                    # invalidation after forced async preemption. Without it,
                    # the model runner can retain a stale persistent-batch id.
                    self.prev_step_scheduled_req_ids.clear()
            requested_defer = {
                str(request_id)
                for request_id in (
                    getattr(decision, "defer_ids", None) or []
                )
            }
            # FCFSRequestQueue subclasses deque -> reorder in place. Any other
            # queue (PriorityRequestQueue heap) is left untouched on purpose.
            if isinstance(self.waiting, deque):
                indexed_reorder_only = (
                    indexed_view
                    and not requested_defer
                    and not type(self)._ve_defer_counts
                    and not preempted
                )
                bounded_prefix_only = (
                    not indexed_view
                    and waiting_window > 0
                    and not requested_defer
                    and not type(self)._ve_defer_counts
                    and not preempted
                )
                if indexed_reorder_only:
                    indexed_by_id = {
                        str(getattr(request, "request_id", "")): request
                        for request in waiting_list
                    }
                    selected = next(
                        (
                            indexed_by_id[str(request_id)]
                            for request_id in decision.prefill_batch
                            if str(request_id) in indexed_by_id
                        ),
                        None,
                    )
                    if selected is not None and selected is not self.waiting[0]:
                        self.waiting.remove(selected)
                        self.waiting.appendleft(selected)
                        _emit(_PLUGIN_REORDERED_MARKER)
                        type(self)._ve_reordered = True
                elif bounded_prefix_only:
                    # The policy saw exactly this prefix and can only return IDs
                    # from it. Reorder those objects without touching,
                    # converting, or rebuilding the unobserved deque tail.
                    current_prefix = list(
                        islice(self.waiting, waiting_window)
                    )
                    prefix_ids = [
                        str(getattr(request, "request_id", ""))
                        for request in current_prefix
                    ]
                    id_to_req = {
                        request_id: request
                        for request_id, request in zip(
                            prefix_ids, current_prefix
                        )
                    }
                    order = _reorder_waiting(
                        prefix_ids, list(decision.prefill_batch)
                    )
                    reordered = [
                        id_to_req[request_id]
                        for request_id in order
                        if request_id in id_to_req
                    ]
                    if reordered != current_prefix:
                        for _ in range(len(current_prefix)):
                            self.waiting.popleft()
                        self.waiting.extendleft(reversed(reordered))
                        _emit(_PLUGIN_REORDERED_MARKER)
                        type(self)._ve_reordered = True
                else:
                    # Compatibility path for legacy whole-queue policies,
                    # active preemption, and explicit bounded deferral.
                    current_waiting = list(self.waiting)
                    id_to_req = {
                        str(getattr(r, "request_id", "")): r
                        for r in current_waiting
                    }
                    forced_admit_ids = []
                    for request in current_waiting:
                        request_id = str(
                            getattr(request, "request_id", "")
                        )
                        if request_id not in requested_defer:
                            type(self)._ve_defer_counts.pop(
                                request_id, None
                            )
                            continue
                        defer_count = (
                            int(
                                type(self)._ve_defer_counts.get(
                                    request_id, 0
                                )
                            )
                            + 1
                        )
                        if defer_count >= type(self)._ve_max_defer_events:
                            type(self)._ve_defer_counts.pop(
                                request_id, None
                            )
                            forced_admit_ids.append(request_id)
                            _emit(_PLUGIN_FORCED_ADMIT_MARKER)
                        else:
                            type(self)._ve_defer_counts[
                                request_id
                            ] = defer_count
                            deferred.append(request)
                            _emit(_PLUGIN_DEFERRED_MARKER)
                    deferred_ids = {
                        str(getattr(request, "request_id", ""))
                        for request in deferred
                    }
                    active_ids = [
                        request_id
                        for request_id in id_to_req
                        if request_id not in deferred_ids
                    ]
                    priority_ids = [
                        *forced_admit_ids,
                        *list(decision.prefill_batch),
                    ]
                    order = _reorder_waiting(active_ids, priority_ids)
                    reordered = [
                        id_to_req[i] for i in order if i in id_to_req
                    ]
                    changed = reordered != current_waiting
                    self.waiting.clear()
                    self.waiting.extend(reordered)
                    if changed:
                        _emit(_PLUGIN_REORDERED_MARKER)
                        type(self)._ve_reordered = True
            if indexed_view:
                type(self)._ve_last_policy_fingerprint = (
                    None if deferred else policy_fingerprint
                )
            else:
                type(self)._ve_last_policy_fingerprint = (
                    None
                    if deferred
                    else policy_fingerprint
                    if allow_active_preemption
                    else tuple(
                        str(getattr(r, "request_id", ""))
                        for r in (
                            islice(self.waiting, waiting_window)
                            if waiting_window
                            else self.waiting
                        )
                    )
                )
        except Exception:
            # L4 fallback: keep vLLM's default ordering on any policy error, but
            # RECORD it (M-B2) so the effective gate can reject this candidate.
            if not type(self)._ve_fallback:
                _emit(_PLUGIN_FALLBACK_MARKER)
                type(self)._ve_fallback = True
            type(self)._ve_last_policy_fingerprint = policy_fingerprint
        scheduler_output = super().schedule()
        if type(self)._ve_global_recoverable_index:
            active = getattr(self, "_ve_index_active", None)
            if active:
                for scheduled in (
                    getattr(scheduler_output, "scheduled_new_reqs", None) or []
                ):
                    active.pop(str(getattr(scheduled, "req_id", "")), None)
        if deferred:
            # A deferred request is hidden for exactly one stock scheduling
            # event, then restored. Repeated deferral is explicit and bounded
            # by _ve_max_defer_events, so omission cannot starve or drop it.
            queued_ids = {
                str(getattr(request, "request_id", ""))
                for request in self.waiting
            }
            running_ids = {
                str(getattr(request, "request_id", ""))
                for request in self.running
            }
            for request in deferred:
                request_id = str(getattr(request, "request_id", ""))
                if request_id not in queued_ids and request_id not in running_ids:
                    self.waiting.append(request)
        if preempted:
            # The model runner keeps a persistent batch in async mode. The
            # stock scheduler includes internally preempted IDs in this field;
            # FTC preempts before delegating, so merge those IDs explicitly.
            if scheduler_output.preempted_req_ids is None:
                scheduler_output.preempted_req_ids = set(preempted)
            else:
                scheduler_output.preempted_req_ids.update(preempted)
        return scheduler_output
'''


def _strip_conflicting_imports(policy_source: str) -> str:
    """Remove imports the template already provides or that won't resolve in the
    container (``from __future__``, ``from targets.*``, Frontier policy API)."""
    out = []
    for line in policy_source.splitlines():
        s = line.strip()
        if s.startswith("from __future__"):
            continue
        if s.startswith("from targets."):
            continue
        if s.startswith("from integrations.frontier.ve_policy_api import"):
            continue
        out.append(line)
    return "\n".join(out)


def render_plugin(policy_source: str) -> str:
    """Return standalone vLLM-V1-plugin module text wrapping ``policy_source``.

    ``policy_source`` must define ``schedule_batch(...)`` returning a
    ScheduleDecision-shaped object with a ``prefill_batch`` list of request ids.
    """
    code = _strip_conflicting_imports(policy_source)
    return _TEMPLATE.replace(_POLICY_MARKER, code)


def render_and_write(plugin_dir: str | Path, policy_source: str) -> Path:
    """Render and write the plugin to ``<plugin_dir>/generated_scheduler.py``.

    The exact filename + top-level placement is REQUIRED: vLLM imports the
    module by the bare name ``generated_scheduler`` from PYTHONPATH.
    """
    path = Path(plugin_dir) / PLUGIN_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_plugin(policy_source))
    return path


# AC-1 / DESIGN naming compatibility.
def wrap_as_vllm_plugin(evolved_source: str, source_info: str = "vllm-evolve",
                        fitness: float = 0.0) -> str:
    return render_plugin(evolved_source)
