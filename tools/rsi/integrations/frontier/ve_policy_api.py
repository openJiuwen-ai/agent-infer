"""Policy-facing decision API for the ve_policy bridge (the optional defer-aware extension).

WHY this is a separate module and not a field added to ``targets/scheduling/skeleton.py``:
the skeleton is a **hook-protected, frozen ground-truth file** — the PreToolUse phase guard denies
edits to ``/skeleton.py`` in every phase, and AC7 requires the safety substrate stay zero-change.
So the plan's optional ``ScheduleDecision.defer_ids`` field is delivered HERE, as a sanctioned,
dependency-free, Frontier-importable surface a policy opts into, rather than by mutating the frozen
contract. ``ScheduleDecision`` below is a strict SUPERSET of the skeleton's (same three fields, plus
the optional ``defer_ids``), so it is backwards compatible: a policy that ignores ``defer_ids``
behaves exactly as before.

NO frontier and NO vllm_evolve imports — pure stdlib only — because a policy runs inside the
SEPARATE Frontier interpreter (``VE_FRONTIER_PYTHON``) where ``vllm_evolve`` is not installed. The
bridge inserts ``<VE_POLICY_ROOT>/integrations/frontier`` on ``sys.path`` (same mechanism as
``ve_defer``); the offline test suite imports it directly. The bridge reads ``prefill_batch`` and
``defer_ids`` off the returned object via ``getattr`` (duck-typed), so this class is a CONVENIENCE
for authoring a defer-aware policy with a stable constructor — not a hard requirement.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScheduleDecision:
    """A schedule_batch result with the optional second admission channel.

    * ``prefill_batch`` — request ids to admit/prefill this step, in policy order.
    * ``decode_batch``  — running request ids to decode this step.
    * ``preempt_ids``   — request ids to preempt (unchanged from the skeleton).
    * ``defer_ids``     — request ids the policy EXPLICITLY moves to the tail of the complete
      waiting view; the bridge's anti-starvation decay force-prioritizes an id deferred for >= N
      consecutive events. Distinct from "forgotten" ids the policy simply never mentions, which
      the bridge FCFS-appends behind ``prefill_batch`` but before explicitly deferred ids.
    """

    prefill_batch: list
    decode_batch: list = field(default_factory=list)
    preempt_ids: list = field(default_factory=list)
    defer_ids: list = field(default_factory=list)
    mechanism_applicable: bool | None = None
