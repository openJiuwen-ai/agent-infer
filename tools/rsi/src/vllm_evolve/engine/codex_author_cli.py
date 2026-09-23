"""Read an AuthorContext on stdin and ask local Codex for one typed child.

This is deliberately a thin transport. The evolution controller owns writing,
static verification, lineage binding, real-vLLM execution, and selection.
Codex runs read-only and can only return a proposal matching the JSON schema.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA = _REPO_ROOT / "config" / "schemas" / "codex_candidate_submission.schema.json"
_BUNDLED_CODEX = Path("/Applications/ChatGPT.app/Contents/Resources/codex")

_AUTHOR_INSTRUCTION = """\
Act as the source-author stage of vllm-evolve's strict real-vLLM evolution loop.
The AuthorContext JSON arrives on stdin as additional context. Return exactly one
JSON object conforming to the supplied output schema; do not edit files.

Author a complete Python source string defining the required schedule_batch
contract from `skeleton` plus only the documented module constants needed by
the bridge (`MAX_DEFER_EVENTS`, `ALLOW_ACTIVE_PREEMPTION`,
`ACTIVE_PREEMPTION_ONLY`, `ACTIVE_PREEMPTION_MIN_OUTPUT_TOKENS`, and
`ACTIVE_PREEMPTION_REQUIRES_FIRST_TOKEN`, `POLICY_WAITING_WINDOW`, and
`POLICY_NEEDS_RUNNING_REQUESTS`, `POLICY_GLOBAL_RECOVERABLE_INDEX`,
`POLICY_INDEX_WIDTH`, and `POLICY_TTFT_SLO_S`). Preserve the exact function
name, argument names, and return annotation. Emit only those constants, optional
policy state/helper functions, and schedule_batch; never copy, redefine, or
shadow the bridge-owned RequestInfo or ScheduleDecision classes. The runtime
RequestInfo exposes truthful
`remaining_output_tokens`, `generated_output_tokens`, `waiting_age_s`,
`has_prefix_hint`, and `session_id`; ScheduleDecision additionally accepts
`defer_ids` and `mechanism_applicable`.

Use only mechanism cards and source IDs present in the frozen research_context.
Choose a structural algorithm whose live upstream gap, trigger, runtime signal,
and ablation match the candidate manifest. In generation 0, use the number and
manifests of same-generation peers to select a distinct mechanism family. In
later generations, start from measured parents and explicitly react to prior
candidate evidence in lessons. A prior candidate with selection_eligible=false
is diagnostic-only: never call its raw_gain_pct a score or treat it as accepted,
but use its source and measurements to preserve useful behavior while repairing
the stated validity failure. Prefer bounded structural mutations that retain the
FCFS-selected workset and its scheduled token fill when prior global reordering
reduced GPU duty, utilization, or memory pressure. Do not emit weighted-score soup, arbitrary
coefficient accumulation, renamed FCFS/SJF, fabricated request IDs, O(n^2)
loops, imports, I/O, network access, or provenance-marker strings.

Treat `operator` as a binding genetic contract. For mutation, transform the one
selected parent's complete source. For crossover, consume exactly the two
distinct verified/scored parents provided in `parents`: inherit a named control-
flow mechanism from each, explain why they are compatible in `structural_change`,
and return one complete executable child. Never splice source text, inherit a
parent score, silently ignore one parent, or claim crossover when the child is
identical to either parent. The orchestrator binds the ordered parent SHAs and
materializes parent-specific ablation controls after submission.

For the new natural-release family, use the bridge's default-off,
arrival-indexed candidate surface. Set POLICY_GLOBAL_RECOVERABLE_INDEX=true,
POLICY_WAITING_WINDOW=0, POLICY_NEEDS_RUNNING_REQUESTS=false,
POLICY_TTFT_SLO_S=120.0, ALLOW_ACTIVE_PREEMPTION=false, and
ACTIVE_PREEMPTION_ONLY=false. The bridge preserves the actual FCFS head and
adds only a bounded union of live candidates from cold-path prompt-cost and
planned-residency heaps. Expired index entries remain untouched in the real
FCFS residual deque; the policy never sees or reconstructs the unobserved
queue. Never call sorted(waiting_requests), never infer whole-queue statistics,
and never use kv_blocks_used because the bridge maps it to zero. Every return
must have empty preempt_ids, defer_ids, and decode_batch. Do not read
running_requests or derive a sequence budget from its length. The bridge keeps
the real running-set capacity guard, moves at most one selected raw request to
deque head, and delegates allocation to unchanged stock AsyncScheduler.

Treat `diagnosis.required_change` as the round-specific design contract. Do not
repeat a mechanism recipe from an earlier round merely because it appears in
this transport instruction. Cite only mechanism cards and source IDs that are
actually present in the frozen `research_context`, and use the diagnosis,
measured parent source/scores, peers, lessons, invalid-run diagnostics, and
remaining budget to choose the concrete controller.

When `diagnosis.required_change.mechanism` is
`closed_loop_prefill_mass_restitution`, implement a Prompt-Mass Restitution
(PMR) controller over the arrival-indexed natural-release surface. For the
production live-gap corpus, combine the compatible frozen cards
`recoverable_short_residency_frontier`,
`slack_guarded_first_token_wavefront`, `adaptive_slo_cliff_switch`, and
`budget_feedback_fill_parity_controller`; add
`decode_mass_anchored_rescue_wave` only when the implementation truly carries
its declared mass ledger. If a different frozen corpus is supplied, select its
semantic equivalents rather than citing absent IDs. This is a stateful feedback
controller, not a renamed FCFS/SJF rule or a fixed rescue ratio:

* Keep bounded nonnegative module state for cumulative displaced prompt mass,
  rescue credit, and a RESCUE/ANCHOR hysteresis mode.
* Filter the bounded visible candidates to positive TTFT slack and live
  prompt-token/KV feasibility. In RESCUE mode, use strict smaller planned
  prompt-plus-output residency before the 80% TTFT cliff and smallest prompt
  cost after the cliff.
* For every real non-head bypass, add
  `max(0, fcfs_head_prompt - selected_prompt)` to prompt-mass debt. Do not admit
  a bypass that would cross the active debt high-water mark.
* At the high-water mark enter ANCHOR mode. Select the exact real FCFS head on
  natural release edges and repay debt by its prompt mass until the low-water
  mark is reached, then return to RESCUE mode. Retain a 63-bypass hard safety
  cap independent of the mass controller.
* When no feasible recoverable candidate exists, select the FCFS head and
  reset rescue credit. Emit exactly one input ID when a waiter exists. A
  no-reorder/head return sets mechanism_applicable=false; a real non-head
  indexed bypass sets it true.

Generation-0 PMR peers must exercise three structurally distinct feedback
surfaces while retaining the same safety contract: no peers -> explicit
high/low-water hysteresis with one live token budget of debt; one peer -> a
prospective debt corridor that vetoes an over-budget rescue before entering a
head-repayment lane; two peers -> a signed mass ledger that also credits
larger-than-head selections and repays to a quarter-budget low watermark. Do
not present these as accepted settings. In later generations, start from the
best measured parent and use valid scores or diagnostic-only validity failures
to alter feedback flow, not merely numeric thresholds.

The algorithm must be work-conserving outside its declared trigger, bound every
priority action, return IDs only from its inputs, and set
mechanism_applicable truthfully. Include `# VE_MECHANISM:<id>` in source for each
cited mechanism. Set manifest proposal_only=true and parameter_only=false. Cite
each selected card's exact source ID as `source:<id>`. candidate_sha,
parent_sha, research_snapshot_hash, author_kind, and generation are bound by the
orchestrator, so placeholders are acceptable for those fields.
"""


def resolve_codex_binary() -> str:
    override = os.environ.get("VE_CODEX_BIN")
    if override:
        return override
    if _BUNDLED_CODEX.is_file():
        return str(_BUNDLED_CODEX)
    discovered = shutil.which("codex")
    if discovered:
        return discovered
    raise FileNotFoundError(
        "Codex CLI not found; set VE_CODEX_BIN or install a working codex binary"
    )


def author_with_codex(
    context_text: str,
    *,
    codex_bin: str | None = None,
    timeout_s: float | None = None,
    run=subprocess.run,
) -> dict:
    context = json.loads(context_text)
    # Legacy exported bundles predate typed genetic operators.  Accept them as an unspecified
    # single-parent author request; every newly rendered AuthorContext includes this field.
    context.setdefault("operator", {})
    required = {
        "doctrine",
        "spec",
        "diagnosis",
        "skeleton",
        "parents",
        "peers",
        "lessons",
        "research_context",
        "last_errors",
        "budget",
        "operator",
        "generation",
        "author_kind",
    }
    missing = sorted(required - set(context))
    if missing:
        raise ValueError(f"AuthorContext is missing fields: {missing}")
    argv = [
        codex_bin or resolve_codex_binary(),
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--cd",
        str(_REPO_ROOT),
        "--output-schema",
        str(_SCHEMA),
    ]
    model = os.environ.get("VE_CODEX_AUTHOR_MODEL")
    if model:
        argv.extend(["--model", model])
    argv.append(_AUTHOR_INSTRUCTION)
    completed = run(
        argv,
        input=context_text,
        text=True,
        capture_output=True,
        timeout=timeout_s
        or float(os.environ.get("VE_CODEX_AUTHOR_TIMEOUT_S", "540")),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Codex author exited {completed.returncode}: {completed.stderr[-2000:]}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Codex author stdout was not one JSON object: {completed.stdout[-1000:]}"
        ) from exc
    if not isinstance(result, dict):
        raise ValueError("Codex author response must be a JSON object")
    if not isinstance(result.get("source"), str) or not result["source"].strip():
        raise ValueError("Codex author response has no source")
    if not isinstance(result.get("manifest"), dict):
        raise ValueError("Codex author response has no manifest")
    return result


def main() -> int:
    try:
        result = author_with_codex(sys.stdin.read())
    except Exception as exc:  # noqa: BLE001 - command contract needs concise stderr
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
