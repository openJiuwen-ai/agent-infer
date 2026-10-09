"""Build a live-pinned scheduler expert snapshot for strict real evolution."""

from __future__ import annotations

from pathlib import Path

from vllm_evolve.knowledge.compiler import CompileResult, live_required_compiler
from vllm_evolve.knowledge.providers import (
    GitHubVLLMUpstreamProvider,
    ResearchRequest,
)


def _gap_cards(main_source_id: str) -> list[dict]:
    common = {
        "applicable_workloads": ["official BurstGPT saturated/high/severe"],
        "applicable_hardware": [
            "two identical NVIDIA GPUs in dual-replica TP=1 mode"
        ],
        "applicable_software": ["vLLM 0.21.0 custom AsyncScheduler"],
        "expected_improved_metrics": [
            "goodput_req_s",
            "TTFT SLO attainment",
            "output_throughput_tok_s",
            "request_throughput_req_s",
        ],
        "possible_regressions": ["TTFT tail", "recomputation", "scheduler overhead"],
        "complexity_cost": "O(n log n) or better per changed waiting frontier",
        "resource_cost": "no model, KV format, or GPU-count change",
        "failure_modes": ["scheduler fallback", "starvation", "preemption thrash"],
        "implementation_hooks": [
            "targets/scheduling/skeleton.py:schedule_batch",
            "targets/scheduling/plugin_template.py:EvolvedScheduler.schedule",
        ],
        "required_controls": [
            "same-caliber stock strong baseline",
            "source-level mechanism ablation",
            "paired held-out seeds 0,1,2",
        ],
        "confidence": "medium",
        "evidence_type": "live_upstream_gap_plus_recent_systems_mechanism",
        "version_constraints": [
            "installed vLLM ad7125a431e176d4161099480a66f0169609a690",
            f"live upstream source {main_source_id}",
        ],
    }
    return [
        {
            **common,
            "mechanism_id": "bounded_first_fit_head_bypass",
            "name": "Bounded first-fit bypass after a KV-unfit queue head",
            "source_ids": [main_source_id, "paper:libra-nsdi26"],
            "problem": (
                "One KV-unfit FCFS head stops stock admission although a later "
                "request may fit the live block and sequence budgets."
            ),
            "core_mechanism": (
                "Temporarily hide exactly one unfit head for one stock scheduling "
                "event and admit the first later request that fits."
            ),
            "assumptions": [
                "waiting queue is deque-backed FCFS",
                "live free KV blocks are visible",
            ],
            "incompatibilities": ["priority-heap request queue"],
            "structural_change": (
                "Replace break-on-head-failure with one-event bounded first-fit bypass."
            ),
            "upstream_gap": (
                "Current Scheduler.schedule breaks when allocate_slots returns None "
                "and does not inspect a later fit request."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                "vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots",
            ],
            "current_vllm_behavior": (
                "The waiting allocation loop preserves FCFS head blocking on KV failure."
            ),
            "candidate_delta": (
                "Defer only the blocking head for one event, prioritize one later "
                "request whose exact prompt block need fits, then restore the head."
            ),
            "why_not_already_integrated": (
                "Pinned current main has no continue-after-allocation-failure or "
                "bounded fit-bypass branch."
            ),
            "required_runtime_signal": (
                "waiting queue >0, KV occupancy high, and mixed prompt block demands"
            ),
            "mechanism_trigger": (
                "FCFS head prompt blocks exceed available KV while a later request fits"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "deferred_request_actions",
                "reorder_action_count",
            ],
            "minimum_meaningful_ablation": (
                "Return the same priority order but remove only defer_ids for the unfit head."
            ),
        },
        {
            **common,
            "mechanism_id": "max_cardinality_kv_fit_frontier",
            "name": "Maximum-cardinality KV-fit admission frontier",
            "source_ids": [main_source_id, "paper:libra-nsdi26"],
            "problem": (
                "Head-local admission does not construct a feasible multi-request "
                "frontier from heterogeneous prompt and decode sizes."
            ),
            "core_mechanism": (
                "Build a work-conserving feasible frontier by ascending exact KV-block "
                "need, admitting the maximum number that fit current token/KV/seq budgets."
            ),
            "assumptions": ["request block demand is observable before admission"],
            "incompatibilities": ["workloads with no waiting-size heterogeneity"],
            "structural_change": (
                "Replace single-head admission with a cardinality-maximizing feasible frontier."
            ),
            "upstream_gap": (
                "Current main has no admission pass that searches heterogeneous waiting "
                "requests for a maximum-cardinality subset after head failure."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                "vllm.v1.core.sched.request_queue.FCFSRequestQueue",
            ],
            "current_vllm_behavior": (
                "FCFS allocation stops at the first request that cannot allocate KV slots."
            ),
            "candidate_delta": (
                "Select a deterministic KV/token-feasible frontier and defer only the "
                "blocking heads that prevent that frontier from reaching stock allocation."
            ),
            "why_not_already_integrated": (
                "No fit-frontier or cardinality search appears in the pinned scheduler source."
            ),
            "required_runtime_signal": (
                "deep waiting queue, high KV occupancy, heterogeneous prompt lengths"
            ),
            "mechanism_trigger": (
                "at least two waiting requests have different KV needs and not all fit"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "deferred_request_actions",
                "reorder_action_count",
            ],
            "minimum_meaningful_ablation": (
                "Preserve KV feasibility checks but restore FCFS order and remove frontier defers."
            ),
        },
        {
            **common,
            "mechanism_id": "one_shot_residual_exchange",
            "name": "One-shot residual-work admission exchange",
            "source_ids": [
                main_source_id,
                "paper:fastserve-nsdi26",
                "paper:sola-mlsys25",
            ],
            "problem": (
                "Allocation-failure recovery chooses no explicit victim using remaining "
                "decode work, so long residual jobs can block a much smaller completion."
            ),
            "core_mechanism": (
                "Exchange one never-preempted running residual-work elephant for one "
                "strictly smaller waiting completion opportunity, then protect resumed work."
            ),
            "assumptions": ["planned and generated output lengths are scheduler-visible"],
            "incompatibilities": ["requests whose remaining output length is unknown"],
            "structural_change": (
                "Add a bounded one-victim residual-work exchange before stock allocation failure."
            ),
            "upstream_gap": (
                "Stock preemption is reactive allocation recovery, not residual-work "
                "victim selection from running and waiting requests."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler._preempt_request",
                "vllm.v1.request.Request.num_preemptions",
            ],
            "current_vllm_behavior": (
                "Victims arise from allocation pressure without comparing remaining outputs."
            ),
            "candidate_delta": (
                "Preempt at most one unpreempted maximum-residual running request only "
                "when a waiting request has strictly less total remaining work."
            ),
            "why_not_already_integrated": (
                "Pinned main contains no remaining_output_tokens or residual_work_victim path."
            ),
            "required_runtime_signal": (
                "full sequence slots, nonempty waiting queue, mixed residual output lengths"
            ),
            "mechanism_trigger": (
                "a never-preempted running maximum residual exceeds the smallest waiting total work"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "preemption_count",
            ],
            "minimum_meaningful_ablation": (
                "Keep all admission ordering but set preempt_ids empty."
            ),
        },
        {
            **common,
            "mechanism_id": "completion_protected_residual_exchange",
            "name": "Completion-protected residual-work exchange",
            "source_ids": [
                main_source_id,
                "paper:fastserve-nsdi26",
                "paper:sola-mlsys25",
            ],
            "problem": (
                "Naive preemption can evict near-completion requests and destroy useful "
                "decode work while trying to improve queue turnover."
            ),
            "core_mechanism": (
                "Partition running requests into a protected completion frontier and "
                "unstarted residual elephants; exchange only from the latter set."
            ),
            "assumptions": [
                "generated output count and num_preemptions are visible",
                "waiting total work is heterogeneous",
            ],
            "incompatibilities": ["all running requests already near completion"],
            "structural_change": (
                "Introduce a completion-protected victim set before one-shot residual exchange."
            ),
            "upstream_gap": (
                "Current preemption recovery has no explicit completion protection based "
                "on generated versus remaining output."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler._preempt_request",
                "vllm.v1.request.Request.output_token_ids",
            ],
            "current_vllm_behavior": (
                "Reactive preemption does not expose a completion-frontier victim partition."
            ),
            "candidate_delta": (
                "Protect requests that have begun producing output; permit one exchange "
                "only from unpreempted, pre-output, high-residual running requests."
            ),
            "why_not_already_integrated": (
                "Pinned main has no generated-output-aware victim partition in Scheduler.schedule."
            ),
            "required_runtime_signal": (
                "full slots with both pre-output elephants and shorter waiting requests"
            ),
            "mechanism_trigger": (
                "an unpreempted pre-output elephant has more residual work than a waiting request"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "preemption_count",
            ],
            "minimum_meaningful_ablation": (
                "Remove only completion protection and allow the same residual "
                "exchange over all running."
            ),
        },
        {
            **common,
            "mechanism_id": "recoverable_short_residency_frontier",
            "name": "Recoverable short-residency admission frontier",
            "source_ids": [
                main_source_id,
                "paper:jitserve-nsdi26",
                "paper:sola-mlsys25",
            ],
            "problem": (
                "Under a hard TTFT SLO, FCFS continues spending scarce admission "
                "capacity on requests that can no longer attain the SLO while "
                "younger, shorter-residency requests remain recoverable."
            ),
            "core_mechanism": (
                "Partition waiting work by positive TTFT slack, build a token-, "
                "KV-, and sequence-feasible frontier from recoverable requests "
                "with the shortest planned residency, and serve expired work only "
                "when no recoverable request fits."
            ),
            "assumptions": [
                "the frozen TTFT SLO is present in research environment",
                "waiting age and planned output length are scheduler-visible",
            ],
            "incompatibilities": [
                "objectives without a request-level TTFT deadline",
            ],
            "structural_change": (
                "Add a deadline-recoverability partition before a residency-aware "
                "feasible admission pass."
            ),
            "upstream_gap": (
                "Current main admits from the configured request queue without "
                "classifying waiting requests by remaining TTFT slack."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                "vllm.v1.request.Request.arrival_time",
                "vllm.v1.request.Request.max_tokens",
            ],
            "current_vllm_behavior": (
                "FCFS queue order is independent of request SLO recoverability "
                "and planned decode residency."
            ),
            "candidate_delta": (
                "Prioritize only capacity-feasible, positive-slack requests by "
                "planned prompt-plus-output residency; append expired requests "
                "after the recoverable frontier without dropping them."
            ),
            "why_not_already_integrated": (
                "Pinned main exposes arrival and token budgets but has no TTFT "
                "recoverability partition or expired-work residual lane."
            ),
            "required_runtime_signal": (
                "deep waiting queue, frozen TTFT SLO, truthful waiting age, and "
                "heterogeneous planned output lengths"
            ),
            "mechanism_trigger": (
                "waiting work spans recoverable and expired TTFT slack classes"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "reorder_action_count",
            ],
            "minimum_meaningful_ablation": (
                "Keep the same residency ordering but remove the recoverable-versus-"
                "expired TTFT partition."
            ),
        },
        {
            **common,
            "mechanism_id": "slack_guarded_first_token_wavefront",
            "name": "Slack-guarded first-token wavefront",
            "source_ids": [
                main_source_id,
                "paper:jitserve-nsdi26",
                "paper:sola-mlsys25",
            ],
            "problem": (
                "Pure FCFS ignores prompt-cost heterogeneity, while pure shortest-"
                "prompt ordering can spend capacity on young requests and abandon "
                "requests approaching the TTFT deadline."
            ),
            "core_mechanism": (
                "Form positive-slack deadline bands, rescue the near-cliff band "
                "first by smallest feasible prompt cost, then use remaining "
                "capacity for the ample-slack band and finally expired work."
            ),
            "assumptions": [
                "the frozen TTFT SLO is present in research environment",
                "remaining prompt tokens and waiting age are truthful",
            ],
            "incompatibilities": [
                "workloads without a TTFT objective",
            ],
            "structural_change": (
                "Replace one global queue order with a deadline-band wavefront and "
                "a capacity-feasible first-token rescue pass."
            ),
            "upstream_gap": (
                "Current main has no request-age SLO bands or first-token rescue "
                "wavefront in Scheduler.schedule."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                "vllm.v1.request.Request.arrival_time",
                "vllm.v1.request.Request.num_computed_tokens",
            ],
            "current_vllm_behavior": (
                "The FCFS admission loop neither estimates TTFT slack nor "
                "reorders feasible requests by prompt cost within an urgency band."
            ),
            "candidate_delta": (
                "Create deadline-relative urgency bands from waiting_age_s and "
                "order only within each band by remaining prompt work."
            ),
            "why_not_already_integrated": (
                "Pinned main contains neither a TTFT SLO input nor an urgency-band "
                "admission branch."
            ),
            "required_runtime_signal": (
                "positive TTFT slack dispersion and heterogeneous prompt lengths"
            ),
            "mechanism_trigger": (
                "a deep queue contains both near-deadline and ample-slack requests"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "reorder_action_count",
            ],
            "minimum_meaningful_ablation": (
                "Preserve the prompt-cost order but collapse all positive-slack "
                "requests into one band."
            ),
        },
        {
            **common,
            "mechanism_id": "slo_yield_per_residency_frontier",
            "name": "SLO-yield per residency frontier",
            "source_ids": [
                main_source_id,
                "paper:jitserve-nsdi26",
                "paper:fastserve-nsdi26",
            ],
            "problem": (
                "At sequence-slot saturation, admitting long planned decodes can "
                "hold a slot beyond many waiting TTFT deadlines even when their "
                "prefills are individually feasible."
            ),
            "core_mechanism": (
                "Within the positive-slack cohort, admit a bounded feasible wave "
                "of low-residency requests so each released slot creates another "
                "SLO-attaining admission opportunity; retain an oldest-request "
                "rescue lane and never preempt active work."
            ),
            "assumptions": [
                "planned output length is visible before admission",
                "waiting age is visible and the TTFT SLO is frozen",
            ],
            "incompatibilities": [
                "unknown or untrusted output budgets",
            ],
            "structural_change": (
                "Add a non-preemptive SLO-yield wave that couples admission order "
                "to future sequence-slot residency."
            ),
            "upstream_gap": (
                "Current main does not use planned decode residency when ordering "
                "waiting requests under a TTFT deadline."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                "vllm.v1.request.Request.max_tokens",
                "vllm.v1.request.Request.arrival_time",
            ],
            "current_vllm_behavior": (
                "FCFS can admit a long-residency request before many short "
                "recoverable requests and does not reserve an age rescue lane."
            ),
            "candidate_delta": (
                "Order a positive-slack, capacity-feasible admission wave by "
                "planned output residency while reserving bounded oldest-first "
                "progress; do not preempt or defer."
            ),
            "why_not_already_integrated": (
                "Pinned main has no SLO-yield frontier combining arrival age and "
                "planned output residency."
            ),
            "required_runtime_signal": (
                "full or near-full sequence slots, deep queue, positive TTFT "
                "slack, and heterogeneous planned outputs"
            ),
            "mechanism_trigger": (
                "recoverable waiting requests have materially different planned "
                "decode residency"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "reorder_action_count",
            ],
            "minimum_meaningful_ablation": (
                "Keep the same positive-slack cohort and rescue lane but restore "
                "FCFS order within the yield wave."
            ),
        },
        {
            **common,
            "mechanism_id": "adaptive_slo_cliff_switch",
            "name": "Adaptive SLO-cliff regime switch",
            "source_ids": [
                main_source_id,
                "paper:sola-mlsys25",
                "paper:jitserve-nsdi26",
            ],
            "problem": (
                "One static priority rule cannot both release slots quickly early "
                "in a burst and rescue feasible first tokens as the queue reaches "
                "the TTFT SLO cliff."
            ),
            "core_mechanism": (
                "Use the oldest positive-slack waiting age as a system-state "
                "signal: in the early regime build a low-residency feasible wave; "
                "near the SLO cliff switch to earliest-deadline, low-prompt-cost "
                "rescue; place expired work in a residual lane."
            ),
            "assumptions": [
                "the TTFT SLO is frozen for the run",
                "waiting age, prompt work, and planned output are visible",
            ],
            "incompatibilities": [
                "objectives with no deadline or no queue-age signal",
            ],
            "structural_change": (
                "Introduce a top-level system-state regime switch between slot-"
                "release and deadline-rescue admission control."
            ),
            "upstream_gap": (
                "Current main uses one configured request-queue discipline and "
                "has no queue-age-driven SLO regime switch."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                "vllm.v1.core.sched.request_queue.FCFSRequestQueue",
                "vllm.v1.request.Request.arrival_time",
            ],
            "current_vllm_behavior": (
                "FCFS applies the same arrival order before and during the TTFT "
                "deadline cliff."
            ),
            "candidate_delta": (
                "Select one of two structurally different feasible admission "
                "passes from queue age relative to the frozen SLO, without "
                "preempting or dropping requests."
            ),
            "why_not_already_integrated": (
                "Pinned main contains no SLO-relative regime state or dual "
                "admission path."
            ),
            "required_runtime_signal": (
                "oldest positive-slack age, frozen TTFT SLO, remaining prompt "
                "tokens, and planned output length"
            ),
            "mechanism_trigger": (
                "the oldest recoverable request crosses the frozen SLO-cliff band"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "reorder_action_count",
            ],
            "minimum_meaningful_ablation": (
                "Freeze the policy in the early low-residency regime and remove "
                "only the SLO-cliff switch."
            ),
        },
        {
            **common,
            "mechanism_id": "workset_preserving_slo_exchange",
            "name": "FCFS-workset-preserving SLO exchange",
            "source_ids": [
                main_source_id,
                "paper:jitserve-nsdi26",
                "paper:sola-mlsys25",
            ],
            "complexity_cost": (
                "O(k) policy work for one fixed bounded FCFS head window; never "
                "sort or scan the full waiting queue"
            ),
            "problem": (
                "Global SLO-aware reordering can improve first-token attainment "
                "while changing the admitted workset enough to reduce GPU duty "
                "and invalidate a saturated comparison."
            ),
            "core_mechanism": (
                "Freeze a bounded FCFS head window as the candidate workset, then "
                "perform one-for-one deadline rescue exchanges only inside that "
                "window; keep its membership, cardinality, and the queue tail "
                "identical to FCFS."
            ),
            "assumptions": [
                "waiting age and remaining prompt work are scheduler-visible",
                "the frozen TTFT SLO is present in the research environment",
            ],
            "incompatibilities": [
                "shallow queues with no heterogeneous deadline slack",
            ],
            "structural_change": (
                "Separate FCFS workset selection from bounded within-workset "
                "deadline ordering so rescue cannot globally replace load-bearing work."
            ),
            "upstream_gap": (
                "Current main exposes one configured request-queue order and has "
                "no fixed-membership SLO exchange stage."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                "vllm.v1.core.sched.request_queue.FCFSRequestQueue",
                "vllm.v1.request.Request.arrival_time",
            ],
            "current_vllm_behavior": (
                "FCFS couples workset membership and order; changing priority can "
                "therefore alter both which requests enter and when they enter."
            ),
            "candidate_delta": (
                "Choose a bounded FCFS prefix first, reorder only that same set by "
                "recoverable TTFT slack and remaining prompt work, then append the "
                "untouched FCFS tail."
            ),
            "why_not_already_integrated": (
                "Pinned main has no policy hook that freezes FCFS prefix membership "
                "before applying a request-SLO rescue exchange."
            ),
            "required_runtime_signal": (
                "deep waiting queue, frozen TTFT SLO, waiting age, remaining "
                "prompt tokens, and current sequence headroom"
            ),
            "mechanism_trigger": (
                "the bounded FCFS head window contains both near-cliff recoverable "
                "requests and lower-urgency requests"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "reorder_action_count",
            ],
            "minimum_meaningful_ablation": (
                "Keep the identical frozen FCFS workset and tail but restore FCFS "
                "order inside the head window."
            ),
        },
        {
            **common,
            "mechanism_id": "decode_mass_anchored_rescue_wave",
            "name": "Decode-mass-anchored rescue wave",
            "source_ids": [
                main_source_id,
                "paper:fastserve-nsdi26",
                "paper:jitserve-nsdi26",
            ],
            "complexity_cost": (
                "O(k) policy work for one fixed bounded FCFS head window; token-"
                "mass accounting is confined to that window"
            ),
            "problem": (
                "A short-residency SLO wave can release sequence slots quickly yet "
                "drain planned decode-token mass from the active prefix, lowering "
                "GPU utilization and extending total completion time."
            ),
            "core_mechanism": (
                "Within a fixed FCFS head workset, pair each recoverable short-prompt "
                "rescue with a high-residual decode anchor and maintain a cumulative "
                "planned-output-token floor relative to the same FCFS prefix."
            ),
            "assumptions": [
                "planned remaining output tokens are truthful before admission",
                "waiting age and the frozen TTFT SLO are scheduler-visible",
            ],
            "incompatibilities": [
                "requests with unknown or untrusted output budgets",
            ],
            "structural_change": (
                "Add a deficit controller over cumulative decode-token mass while "
                "constructing an alternating rescue-and-anchor admission prefix."
            ),
            "upstream_gap": (
                "Current main neither uses planned output length in FCFS admission "
                "nor preserves a decode-work floor during deadline-aware reordering."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                "vllm.v1.request.Request.max_tokens",
                "vllm.v1.request.Request.arrival_time",
            ],
            "current_vllm_behavior": (
                "The waiting queue has no cumulative output-mass invariant tying "
                "SLO rescue choices to a load-bearing decode anchor."
            ),
            "candidate_delta": (
                "Use only members of a bounded FCFS prefix; emit an urgent "
                "low-prompt request when recoverable, then an output-heavy anchor "
                "whenever needed to meet the FCFS cumulative decode-mass floor."
            ),
            "why_not_already_integrated": (
                "Pinned main has no admission-prefix token-mass accounting or "
                "rescue/anchor pairing control flow."
            ),
            "required_runtime_signal": (
                "deep waiting queue, heterogeneous prompt and planned output "
                "lengths, waiting age, and frozen TTFT SLO"
            ),
            "mechanism_trigger": (
                "a fixed FCFS head workset contains both a recoverable low-prompt "
                "request and a distinct high-residual decode anchor"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "reorder_action_count",
            ],
            "minimum_meaningful_ablation": (
                "Keep the same workset and SLO rescue ordering but remove the "
                "cumulative decode-mass floor and anchor branch."
            ),
        },
        {
            **common,
            "mechanism_id": "budget_feedback_fill_parity_controller",
            "name": "Budget-feedback fill-parity controller",
            "source_ids": [
                main_source_id,
                "paper:sola-mlsys25",
                "paper:libra-nsdi26",
            ],
            "complexity_cost": (
                "O(k) policy work for one fixed bounded FCFS head window; parity "
                "uses only that window and visible scalar budgets"
            ),
            "problem": (
                "A static rescue quota can over-reorder during ramp or drain and "
                "underfill the engine even though live sequence, token, and KV "
                "budgets show whether FCFS can sustain a full admission frontier."
            ),
            "core_mechanism": (
                "Recompute the FCFS-feasible frontier from visible sequence, prompt-"
                "token, and KV budgets every scheduling event; enable a bounded SLO "
                "rescue quota only when the reordered frontier has parity in request "
                "count and scheduled prompt-token fill, otherwise emit exact FCFS."
            ),
            "assumptions": [
                "available KV blocks and scheduler token/sequence budgets are visible",
                "remaining prompt tokens and waiting age are truthful",
            ],
            "incompatibilities": [
                "bridges that cannot expose live admission budgets",
            ],
            "structural_change": (
                "Introduce a closed-loop fill-parity gate around the rescue branch "
                "instead of applying one static priority rule at every queue state."
            ),
            "upstream_gap": (
                "Current main applies its configured queue discipline without a "
                "candidate-versus-FCFS frontier parity test or SLO rescue controller."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                "vllm.v1.core.kv_cache_manager.KVCacheManager.get_usage",
                "vllm.v1.request.Request.num_computed_tokens",
            ],
            "current_vllm_behavior": (
                "Stock admission consumes live budgets in queue order but does not "
                "compare an alternate SLO frontier against FCFS fill before applying it."
            ),
            "candidate_delta": (
                "Build FCFS and rescue frontiers over the same bounded head set; "
                "commit the rescue order only when its admitted count and prompt-"
                "token fill are no lower, otherwise return the untouched FCFS order."
            ),
            "why_not_already_integrated": (
                "Pinned main has no shadow frontier, fill-parity predicate, or "
                "feedback-controlled rescue/fallback branch."
            ),
            "required_runtime_signal": (
                "live sequence headroom, max batched tokens, available KV blocks, "
                "remaining prompt work, waiting age, and frozen TTFT SLO"
            ),
            "mechanism_trigger": (
                "a deep queue offers an SLO rescue exchange whose shadow frontier "
                "matches or exceeds FCFS request-count and prompt-token fill"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "reorder_action_count",
            ],
            "minimum_meaningful_ablation": (
                "Keep the same rescue frontier but force the parity gate open, "
                "removing only feedback fallback to exact FCFS."
            ),
        },
        {
            **common,
            "mechanism_id": "lifecycle_gated_one_shot_rescue",
            "name": "Recomputation-bounded FCFS-head escrow rescue",
            "source_ids": [
                main_source_id,
                "paper:fastserve-nsdi26",
            ],
            "problem": (
                "When every sequence slot is occupied, many queued requests can "
                "miss their TTFT SLO while long planned decodes retain slots; a "
                "general queue-policy callback also adds unacceptable deep-queue "
                "overhead even when it rarely changes order."
            ),
            "core_mechanism": (
                "On the first-output lifecycle edge, preempt at most one previously "
                "unpreempted request from a high-decode-budget class whose computed "
                "KV span covers the original FCFS head's prompt. Escrow that original "
                "head as the sole priority prefix so the preempted victim cannot "
                "reclaim the slot when vLLM prepends it, then make the victim "
                "permanently ineligible for another active preemption."
            ),
            "assumptions": [
                "planned decode budget and first-output state are truthful",
                "vLLM prepends a preempted request to its waiting queue",
                "the bridge bypasses waiting-queue materialization when no eligible "
                "high-decode-budget victim is running",
                "one bounded prompt recomputation can create a useful admission slot",
            ],
            "incompatibilities": [
                "bridges without active running-to-waiting preemption",
                "workloads with no response-length heterogeneity",
            ],
            "structural_change": (
                "Add an atomic running-to-waiting transition plus original-head "
                "escrow: match the smallest recomputation-debt victim that covers "
                "the FCFS head prompt, preempt it once, and place only that original "
                "head ahead of the victim vLLM prepends."
            ),
            "upstream_gap": (
                "Current main preempts on allocation failure but has no planned-"
                "decode-budget class, first-output trigger, recomputation-dominance "
                "match, or original-head escrow for proactive TTFT admission rescue."
            ),
            "upstream_symbols_checked": [
                "vllm.v1.core.sched.scheduler.Scheduler.schedule",
                "vllm.v1.core.sched.scheduler.Scheduler._preempt_request",
                "vllm.v1.request.Request.max_tokens",
                "vllm.v1.request.Request.num_preemptions",
            ],
            "current_vllm_behavior": (
                "Stock AsyncScheduler keeps running requests until completion or "
                "reactive KV-allocation failure. Its preemption path prepends the "
                "victim, so preemption alone normally gives the released slot back "
                "to the same request instead of the pre-existing FCFS head."
            ),
            "candidate_delta": (
                "Enable the bridge's active-preemption-only mode, require a first "
                "output token, select one num_preemptions==0 victim above a frozen "
                "decode-budget threshold whose num_computed_tokens covers the "
                "original head prompt, prioritize exactly that original FCFS head "
                "ahead of the requeued victim, and delegate all ineligible events "
                "to stock scheduling."
            ),
            "why_not_already_integrated": (
                "Pinned current main exposes the needed lifecycle, decode-budget, "
                "and computed-token signals but contains no proactive one-shot "
                "head-escrow exchange controller."
            ),
            "required_runtime_signal": (
                "nonempty waiting queue, full sequence slots, original FCFS head "
                "prompt work, and a first-token num_preemptions==0 running request "
                "above the decode threshold whose computed span covers that prompt"
            ),
            "mechanism_trigger": (
                "the original FCFS head is waiting and a first-token high-decode "
                "victim can cover its prompt with bounded recomputation debt"
            ),
            "expected_action_counters": [
                "policy_invocation_count",
                "preemption_count",
                "reorder_action_count",
            ],
            "minimum_meaningful_ablation": (
                "Keep the identical original-head escrow, decode-budget "
                "classification, dominance guard, and victim selection but disable "
                "only preempt_ids, so the running-to-waiting exchange disappears."
            ),
            "complexity_cost": (
                "O(r) cold eligibility scan over running requests, O(1) inspection "
                "of the original FCFS head, and one single-ID priority prefix; no "
                "waiting-tail scan, sort, defer, or global priority score"
            ),
            "resource_cost": (
                "bounded prompt recomputation for each distinct one-shot victim; "
                "no model, KV format, or GPU-count change"
            ),
            "failure_modes": [
                "excess prompt recomputation",
                "too-sparse rescue class",
                "too-broad rescue class",
                "inactive-path overhead",
                "scheduler fallback",
            ],
            "implementation_hooks": [
                "targets/scheduling/skeleton.py:schedule_batch",
                "targets/scheduling/plugin_template.py:EvolvedScheduler.schedule",
                "targets/scheduling/plugin_template.py:EvolvedScheduler._preempt_request",
            ],
        },
    ]


def _select_gap_cards(main_source_id: str, environment: dict) -> list[dict]:
    falsified_ids = {
        str(item.get("mechanism_id"))
        for item in environment.get("falsified_hypotheses") or []
        if isinstance(item, dict) and item.get("mechanism_id")
    }
    return [
        card
        for card in _gap_cards(main_source_id)
        if str(card.get("mechanism_id")) not in falsified_ids
    ]


def build_real_scheduler_research(
    *,
    out_dir: str | Path,
    installed_vllm_commit: str,
    environment: dict | None = None,
    spec: dict | None = None,
    diagnosis: dict | None = None,
    refresh: bool = True,
) -> CompileResult:
    supplied_environment = dict(environment or {})
    primary_metric = str(
        supplied_environment.get("primary_metric") or "goodput_req_s"
    )
    resolved_spec = {
        "metric": primary_metric,
        "direction": "max",
        "raw_intent": (
            f"evolve structural scheduler algorithms for {primary_metric} "
            "on saturated official BurstGPT"
        ),
        **dict(spec or {}),
    }
    resolved_diagnosis = {
        "bottleneck": "scheduling_queue_under_kv_pressure",
        "status": "confirmed_by_real_calibration",
        **dict(diagnosis or {}),
    }
    base_environment = {
        "source": "real_vllm",
        "workloads": ["official BurstGPT saturated/high/severe"],
        "workload_signals": [],
        "installed_vllm_commit": installed_vllm_commit,
        "vllm_version": "0.21.0",
        "strong_baseline": {
            "continuous_batching": True,
            "async_scheduling": True,
            "chunked_prefill": True,
            "prefix_caching": False,
            "preemption": True,
        },
        **supplied_environment,
    }
    request = ResearchRequest(
        target="scheduling",
        spec=resolved_spec,
        diagnosis=resolved_diagnosis,
        environment=base_environment,
        queries=["current vLLM scheduler live gaps"],
    )
    upstream = GitHubVLLMUpstreamProvider().fetch(request)
    if upstream.status not in {"ok", "partial"}:
        raise RuntimeError(f"live upstream discovery failed: {upstream.detail}")
    main_source = next(
        (
            source.source_id
            for source in upstream.sources
            if source.source_id.startswith("github:vllm-main:")
        ),
        None,
    )
    if not main_source:
        raise RuntimeError("live upstream discovery returned no pinned main source")
    base_environment["live_gap_cards"] = _select_gap_cards(
        main_source,
        base_environment,
    )
    return live_required_compiler().compile(
        target="scheduling",
        spec=resolved_spec,
        diagnosis=resolved_diagnosis,
        environment=base_environment,
        out_dir=out_dir,
        internal_evidence={},
        refresh=refresh,
    )


__all__ = ["build_real_scheduler_research"]
