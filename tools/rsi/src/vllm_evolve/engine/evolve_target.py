"""``evolve_fn`` implementations for autopt's ``code:schedule_batch`` target.

* ``default_evolve_fn`` — honest no-op for the REAL backend (the legacy GA engine was removed;
  real code-surface evolution is box-gated future work).
* ``sim_evolve_fn`` — Phase D: deterministic policy variants evaluated INSIDE the Frontier
  simulator (the ve_policy bridge executes each variant for real). Sim scores are a SEARCH
  signal only: they rank variants and produce a sim-winner PROPOSAL; ``marker_verified`` comes
  from the bench profile (always False under frontier_sim), so the orchestrator can never adopt
  a sim-ranked policy — adoption still requires real vLLM.
"""

from __future__ import annotations

import json

from vllm_evolve.core.schemas import Candidate, Profile, Spec


def _unverified(note: str) -> Candidate:
    return Candidate(
        target="code:schedule_batch", kind="code", value={}, marker_verified=False, note=note
    )


def default_evolve_fn(
    base_config: dict, spec: Spec, *, target: str = "scheduling", **_kw
) -> Candidate:
    """Honest no-op evolver: returns an UNVERIFIED candidate (the legacy GA engine was removed)."""
    return _unverified(
        "code:schedule_batch evolution is not yet wired to the real backend (legacy GA engine "
        "removed); autopt does config-search only — never adopts an unproven policy"
    )


# ── Evolution v2: a COMPOSITIONAL policy grammar (real structure, not just a sort key) ──
# A variant = (order x gate x switch). `order` is the prefill sort key; `gate` adds a real
# admission rule through bridge-v2's explicit ``defer_ids`` channel; `switch` adds a top-level
# LOAD-ADAPTIVE branch that changes order only under queue pressure. So the search produces
# STRUCTURALLY different schedulers — burst-aware admission, total-work ordering, anti-starvation,
# and load switching — rather than five textbook sort orderings. Every recipe is a complete
# policy that clears the SAME full L1/L2 verify as a hand-authored one (one admission loop, so the
# nested-loop heuristic never trips). A `VE_STRUCT:` tag lets feedback build on a parent's STRUCTURE
# (not just a scalar), so later generations explore structural neighbours of the best.

_ORDER = {
    "fcfs": "r.arrival_time_s",
    "sjf": "(r.remaining_prompt_tokens, r.arrival_time_s)",
    "ljf": "(-r.remaining_prompt_tokens, r.arrival_time_s)",
    "lifo": "-r.arrival_time_s",
    "total_work": "(r.remaining_prompt_tokens + r.num_output_tokens, r.arrival_time_s)",
    "cache_first": "(-r.prefix_cached_tokens, r.remaining_prompt_tokens)",
    "starvation": "(-r.num_preemptions, r.remaining_prompt_tokens, r.arrival_time_s)",
}
_GATES = ("none", "burst_tail", "burst_half", "seq_reserve")

_POLICY_TEMPLATE = '''"""{name} schedule_batch variant (sim-search candidate; generated). {tag}"""
from __future__ import annotations
from integrations.frontier.ve_policy_api import ScheduleDecision
from targets.scheduling.skeleton import RequestInfo


def schedule_batch(
    waiting_requests: list[RequestInfo],
    running_requests: list[RequestInfo],
    max_num_batched_tokens: int,
    max_num_seqs: int,
    available_kv_blocks: int,
    prefix_cache_hit_rate: float,
) -> ScheduleDecision:
    decode_batch = [r.request_id for r in running_requests if not r.is_prefill]
    token_budget = max_num_batched_tokens - sum(
        r.remaining_prompt_tokens for r in running_requests if r.is_prefill
    )
    seq_budget = max_num_seqs - len(running_requests)
{order_block}
{gate_block}
    deferred = set(defer_ids)
    prefill_batch = []
    for req in sorted(waiting_requests, key=_order):
        if req.request_id in deferred:
            continue
        need = req.remaining_prompt_tokens
        if need <= token_budget and len(prefill_batch) < seq_budget:
            prefill_batch.append(req.request_id)
            token_budget -= need
    return ScheduleDecision(
        prefill_batch=prefill_batch,
        decode_batch=decode_batch,
        defer_ids=defer_ids,
    )
'''


def _build_policy(
    name: str,
    *,
    order: str = "sjf",
    gate: str = "none",
    switch: bool = False,
    pressure_order: str = "total_work",
    dispersion_guard: bool = False,
    mechanism_id: str = "",
    mechanism_ids: list[str] | None = None,
    research_inspiration_id: str = "",
    research_inspiration_ids: list[str] | None = None,
) -> str:
    """Render a complete schedule_batch policy from the (order, gate, switch) grammar."""
    order_expr = _ORDER.get(order, _ORDER["sjf"])
    if switch:  # total-work order only under system pressure
        pressure_expr = _ORDER.get(pressure_order, _ORDER["total_work"])
        if dispersion_guard:
            order_block = (
                "    queue_pressure = len(waiting_requests) + len(running_requests)\n"
                "    outputs = sorted(r.num_output_tokens for r in waiting_requests)\n"
                "    median_output = outputs[len(outputs) // 2] if outputs else 0\n"
                "    has_prefix_signal = any(\n"
                "        bool(getattr(r, 'has_prefix_hint', False))\n"
                "        for r in waiting_requests\n"
                "    )\n"
                "    mixed_decode_modes = bool(\n"
                "        outputs and 4 * outputs[0] <= max(1, median_output)\n"
                "    )\n"
                "    if queue_pressure <= max(1, max_num_seqs):\n"
                f"        _order = lambda r: {order_expr}\n"
                "    elif mixed_decode_modes and not has_prefix_signal:\n"
                "        _order = lambda r: "
                "(-r.remaining_prompt_tokens, r.arrival_time_s)\n"
                "    else:\n"
                f"        _order = lambda r: {pressure_expr}"
            )
        else:
            order_block = (
                "    queue_pressure = len(waiting_requests) + len(running_requests)\n"
                "    if queue_pressure > max(1, max_num_seqs):\n"
                f"        _order = lambda r: {pressure_expr}\n"
                "    else:\n"
                f"        _order = lambda r: {order_expr}"
            )
    else:
        order_block = f"    _order = lambda r: {order_expr}"

    if gate == "burst_tail":
        gate_block = (
            "    defer_ids = []\n"
            "    if len(waiting_requests) > max(4, max_num_seqs):\n"
            "        by_work = sorted(\n"
            "            waiting_requests,\n"
            "            key=lambda r: "
            "(r.remaining_prompt_tokens + r.num_output_tokens, r.arrival_time_s),\n"
            "        )\n"
            "        keep = max(1, (len(by_work) * 3) // 4)\n"
            "        defer_ids = [r.request_id for r in by_work[keep:]]"
        )
    elif gate == "burst_half":
        gate_block = (
            "    defer_ids = []\n"
            "    if len(waiting_requests) > max(4, max_num_seqs):\n"
            "        by_work = sorted(\n"
            "            waiting_requests,\n"
            "            key=lambda r: "
            "(r.remaining_prompt_tokens + r.num_output_tokens, r.arrival_time_s),\n"
            "        )\n"
            "        keep = max(1, len(by_work) // 2)\n"
            "        defer_ids = [r.request_id for r in by_work[keep:]]"
        )
    elif gate == "seq_reserve":
        gate_block = (
            "    defer_ids = []\n"
            "    if len(waiting_requests) > max(4, max_num_seqs):\n"
            "        by_work = sorted(\n"
            "            waiting_requests,\n"
            "            key=lambda r: "
            "(r.remaining_prompt_tokens + r.num_output_tokens, r.arrival_time_s),\n"
            "        )\n"
            "        keep = max(1, max_num_seqs - len(running_requests))\n"
            "        defer_ids = [r.request_id for r in by_work[keep:]]"
        )
    else:
        gate_block = "    defer_ids = []"
    tag = (
        f"VE_STRUCT:order={order};gate={gate};switch={int(switch)};"
        f"dispersion={int(dispersion_guard)};pressure_order={pressure_order}"
    )
    tagged_mechanisms = list(dict.fromkeys([*(mechanism_ids or []), mechanism_id]))
    for item in tagged_mechanisms:
        if item:
            tag += f" VE_MECHANISM:{item}"
    inspirations = list(
        dict.fromkeys([*(research_inspiration_ids or []), research_inspiration_id])
    )
    for item in inspirations:
        if item:
            tag += f" VE_RESEARCH_INSPIRATION:{item}"
    return _POLICY_TEMPLATE.format(
        name=name, tag=tag, order_block=order_block, gate_block=gate_block
    )


# Seed catalogue: textbook controls plus structurally distinct schedulers.  LACQ is the primary
# algorithm seed: FCFS at low load, completion-aware ordering under pressure. It is a
# regime-switching queue that accounts for decode slot residency, not a renamed prompt sort key.
_CATALOG: list[tuple[str, dict]] = [
    ("fcfs", {"order": "fcfs"}),
    ("sjf", {"order": "sjf"}),
    ("ljf", {"order": "ljf"}),
    ("lifo", {"order": "lifo"}),
    (
        "d3q",
        {
            "order": "fcfs",
            "gate": "none",
            "switch": True,
            "dispersion_guard": True,
        },
    ),
    ("lacq", {"order": "fcfs", "gate": "none", "switch": True}),
    ("burst_half", {"order": "total_work", "gate": "burst_half", "switch": True}),
    ("seq_reserve", {"order": "total_work", "gate": "seq_reserve", "switch": True}),
    ("cache_first", {"order": "cache_first"}),  # prefix-cache-aware ordering
    ("starvation", {"order": "starvation"}),  # anti-starvation (preempt-aware)
    ("load_adaptive", {"order": "sjf", "gate": "none", "switch": True}),
]


def generate_variants(out_dir) -> list[tuple[str, str, str]]:
    """Write the structural variant policy files; return [(name, path, sha256)]."""
    import hashlib
    import os

    os.makedirs(out_dir, exist_ok=True)
    out = []
    for name, kw in _CATALOG:
        src = _build_policy(name.upper(), **kw)
        path = os.path.join(out_dir, f"variant_{name}.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        out.append((name, path, hashlib.sha256(src.encode()).hexdigest()))
    return out


# ── Evolution v2: the headless author (deterministic, feedback-consuming, testable) ──────────
#
# Same AuthorContext contract as the ve-author agent; returns a SOURCE STRING (never writes files).
# Generation 0 seeds the STRUCTURAL catalogue; later generations read the best parent's VE_STRUCT
# recipe and mutate ONE dimension (order / gate / switch), so feedback shapes STRUCTURE, not just a
# scalar. A verify error falls back to the always-valid SJF seed.


def _parse_struct(source: str) -> dict:
    import re

    m = re.search(
        r"VE_STRUCT:order=(\w+);gate=(\w+);switch=([01])"
        r"(?:;dispersion=([01]))?(?:;pressure_order=(\w+))?",
        source or "",
    )
    if not m:
        return {
            "order": "sjf",
            "gate": "none",
            "switch": False,
            "dispersion_guard": False,
            "pressure_order": "total_work",
        }
    return {
        "order": m.group(1),
        "gate": m.group(2),
        "switch": m.group(3) == "1",
        "dispersion_guard": m.group(4) == "1",
        "pressure_order": m.group(5) or "total_work",
    }


def _mutate_recipe(recipe: dict, slot: int) -> dict:
    """Deterministically change ONE structural dimension to explore a neighbour of the parent."""
    r = dict(recipe)
    dim = slot % 4
    if dim == 0:
        orders = list(_ORDER)
        i = orders.index(r["order"]) if r.get("order") in orders else 0
        r["order"] = orders[(i + 1) % len(orders)]
    elif dim == 1:
        i = _GATES.index(r["gate"]) if r.get("gate") in _GATES else 0
        r["gate"] = _GATES[(i + 1) % len(_GATES)]
    elif dim == 2:
        r["switch"] = not r.get("switch", False)
    else:
        r["dispersion_guard"] = not r.get("dispersion_guard", False)
    return r


def _crossover_recipe(primary: dict, secondary: dict, slot: int) -> dict | None:
    """Compose typed recipe fields, never source text, and inherit from both parents.

    Rotating masks provide deterministic alternatives for repair/dedup.  A recipe is accepted only
    when it differs from both parents and at least one differing field came from each.  If the pair
    is structurally degenerate the caller must downgrade honestly instead of claiming crossover.
    """
    fields = ("order", "gate", "switch", "pressure_order", "dispersion_guard")
    for attempt in range(len(fields)):
        offset = (slot + attempt) % len(fields)
        child = {
            field: (primary if (index + offset) % 2 == 0 else secondary).get(field)
            for index, field in enumerate(fields)
        }
        if child == primary or child == secondary:
            continue
        from_primary = any(
            primary.get(field) != secondary.get(field) and child.get(field) == primary.get(field)
            for field in fields
        )
        from_secondary = any(
            primary.get(field) != secondary.get(field) and child.get(field) == secondary.get(field)
            for field in fields
        )
        if from_primary and from_secondary:
            return child
    return None


def _recipe_name(r: dict) -> str:
    suffix = "-SW" if r["switch"] else ""
    suffix += "-D3" if r.get("dispersion_guard") else ""
    return f"{r['order'].upper()}-{r['gate']}{suffix}"


def _parse_mechanisms(source: str) -> list[str]:
    import re

    return list(
        dict.fromkeys(re.findall(r"VE_MECHANISM:([A-Za-z0-9_.-]+)", source or ""))
    )


def _parse_research_inspirations(source: str) -> list[str]:
    import re

    return list(
        dict.fromkeys(
            re.findall(r"VE_RESEARCH_INSPIRATION:([A-Za-z0-9_.-]+)", source or "")
        )
    )


def _parse_mechanism(source: str) -> str:
    """Legacy primary-mechanism helper retained for callers outside the genetic engine."""
    mechanisms = _parse_mechanisms(source)
    return mechanisms[0] if mechanisms else ""


def template_author_fn(ctx) -> str:
    """Headless author consuming the same complete prompt contract as a Codex author."""
    prompt = json.loads(ctx.to_prompt())
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
    missing = required - prompt.keys()
    if missing:
        raise ValueError(f"AuthorContext prompt missing required fields: {sorted(missing)}")
    # Access every feedback channel here, even though the deterministic fallback grammar currently
    # uses only peers/errors/parents/spec to choose a recipe. A Codex completion consumes this exact
    # same JSON and may use diagnosis, skeleton, lessons, budget, and generation semantically.
    diagnosis = prompt["diagnosis"]
    skeleton = prompt["skeleton"]
    lessons = prompt["lessons"]
    research_context = prompt["research_context"]
    budget = prompt["budget"]
    generation = prompt["generation"]
    if not isinstance(diagnosis, dict) or not isinstance(skeleton, str):
        raise ValueError("invalid diagnosis or skeleton in AuthorContext prompt")
    if (
        not isinstance(lessons, list)
        or not isinstance(budget, dict)
        or not isinstance(research_context, dict)
    ):
        raise ValueError("invalid lessons or budget in AuthorContext prompt")
    if not isinstance(generation, int):
        raise ValueError("invalid generation in AuthorContext prompt")

    peers = prompt["peers"]
    parents = prompt["parents"]
    last_errors = prompt["last_errors"]
    spec = prompt["spec"]
    operator = prompt["operator"]
    slot = len(peers)  # which child of this generation
    operator_kind = str(operator.get("kind") or "") if isinstance(operator, dict) else ""
    if last_errors:  # repair the selected parent's typed recipe when one exists
        if parents:
            base = _parse_struct(parents[0].get("source", ""))
            repaired = _mutate_recipe(base, slot + int(operator.get("repair_attempt", 1)))
            return _build_policy(
                f"{_recipe_name(repaired)}-REPAIR",
                mechanism_ids=_parse_mechanisms(parents[0].get("source", "")),
                research_inspiration_ids=_parse_research_inspirations(
                    parents[0].get("source", "")
                ),
                **repaired,
            )
        return _build_policy("SJF-REPAIR", order="sjf")
    scored = [p for p in parents if p.get("score") is not None]
    if not scored and research_context.get("mechanism_cards"):
        # Research-guided generation 0: cover different mechanism families.  The curated
        # compiler maps each mechanism to the nearest deterministic template recipe, while
        # Codex authors receive the complete card and may implement beyond this grammar.
        cards = [
            card
            for card in research_context["mechanism_cards"]
            if isinstance(card, dict) and isinstance(card.get("template_recipe"), dict)
        ]
        portfolio_ids = [
            mechanism_id
            for row in research_context.get("recommended_hypothesis_portfolio", [])
            if isinstance(row, dict) and row.get("status") == "candidate"
            for mechanism_id in row.get("mechanism_ids", [])
        ]
        if portfolio_ids:
            by_id = {card.get("mechanism_id"): card for card in cards}
            cards = [by_id[item] for item in portfolio_ids if item in by_id]
        if cards:
            card = cards[slot % len(cards)]
            allowed = {
                key: value
                for key, value in card["template_recipe"].items()
                if key
                in {
                    "order",
                    "gate",
                    "switch",
                    "pressure_order",
                    "dispersion_guard",
                }
            }
            return _build_policy(
                card.get("name", card["mechanism_id"]).upper(),
                # A deterministic grammar recipe is only research-inspired.  It must never claim
                # semantic equivalence to the paper mechanism in VE_MECHANISM or the manifest.
                research_inspiration_id=card["mechanism_id"],
                **allowed,
            )
    if not scored:  # gen 0: seed the structural catalogue
        name, kw = _CATALOG[slot % len(_CATALOG)]
        return _build_policy(name.upper(), **kw)
    if operator_kind == "crossover" and len(scored) == 2:
        primary, secondary = scored
        crossed = _crossover_recipe(
            _parse_struct(primary.get("source", "")),
            _parse_struct(secondary.get("source", "")),
            slot,
        )
        if crossed is not None:
            mechanisms = list(
                dict.fromkeys(
                    [
                        *_parse_mechanisms(primary.get("source", "")),
                        *_parse_mechanisms(secondary.get("source", "")),
                    ]
                )
            )
            inspirations = list(
                dict.fromkeys(
                    [
                        *_parse_research_inspirations(primary.get("source", "")),
                        *_parse_research_inspirations(secondary.get("source", "")),
                    ]
                )
            )
            return _build_policy(
                f"{_recipe_name(crossed)}-CROSSOVER",
                mechanism_ids=mechanisms,
                research_inspiration_ids=inspirations,
                **crossed,
            )
        # The recipes cannot express a child that inherits a distinct component from both.  Return
        # a complete typed mutation; the verifier will reject a false crossover lineage and feed it
        # back through the repair channel rather than accepting a source splice.
        mutated = _mutate_recipe(_parse_struct(primary.get("source", "")), slot)
        return _build_policy(
            f"{_recipe_name(mutated)}-CROSSOVER-DEGENERATE",
            mechanism_ids=_parse_mechanisms(primary.get("source", "")),
            research_inspiration_ids=_parse_research_inspirations(
                primary.get("source", "")
            ),
            **mutated,
        )
    # later generations: mutate the explicitly selected parent. Legacy contexts without an
    # operator retain the historical best-parent behavior.
    minimize = (spec or {}).get("direction") == "min"
    best = scored[0] if operator_kind in {"mutation", "repair"} else (
        min if minimize else max
    )(scored, key=lambda p: p["score"])
    mutated = _mutate_recipe(_parse_struct(best.get("source", "")), slot)
    return _build_policy(
        _recipe_name(mutated),
        mechanism_ids=_parse_mechanisms(best.get("source", "")),
        research_inspiration_ids=_parse_research_inspirations(best.get("source", "")),
        **mutated,
    )


template_author_fn.ve_author_kind = "template"
template_author_fn.ve_require_manifest = False


def sim_evolve_fn(
    profile_fn, *, variants_dir: str = "runs/evolve_variants", winner_dir: str = "runs/sim_winner"
):
    """Phase D evolve_fn: generate deterministic variants, L1/L2-verify each, evaluate each INSIDE
    the simulator (``profile_fn`` = collect_profile_frontier; the ve_policy bridge really executes
    the policy), rank by the spec metric — SEARCH ONLY. The returned Candidate's
    ``marker_verified`` comes from the bench profile (False under frontier_sim), so the
    orchestrator records ``no_verified_candidate`` and can never adopt a sim-ranked policy.
    The best variant is written out as a sim-winner PROPOSAL with an explicit
    REQUIRES_REAL_VLLM_VERIFICATION stamp."""
    import json
    import os
    import shutil

    from vllm_evolve.trust.safety import check_safety, check_signatures

    def evolve_fn(base_config: dict, spec: Spec) -> Candidate:
        trials: list[dict] = []
        best = None  # (score, name, path, sha, prof)
        evals = 0
        for name, path, sha in generate_variants(variants_dir):
            src = open(path, encoding="utf-8").read()
            safety = check_safety(src)
            sig = check_signatures(src, ["schedule_batch"])
            if not (safety.ok and sig.ok):
                trials.append(
                    {
                        "value": {"policy": path},
                        "sim_score": None,
                        "verified": False,
                        "note": "failed L1/L2 verify",
                    }
                )
                continue
            prof = profile_fn({**base_config, "runner_kind": "candidate", "policy": path})
            evals += 1
            score = None
            if isinstance(prof, Profile):
                score = prof.metrics.get(spec.metric)
                if score is None and prof.metrics:
                    score = next(iter(prof.metrics.values()))
            trials.append(
                {
                    "value": {"policy": path},
                    "sim_score": score,
                    "policy_sha": sha,
                    "verified": False,
                    "note": "sim score — search-only, never adoption",
                }
            )
            if score is not None:
                better = (
                    best is None
                    or (spec.direction == "min" and score < best[0])
                    or (spec.direction != "min" and score > best[0])
                )
                if better:
                    best = (score, name, path, sha, prof)
        if best is None:
            return _unverified("sim evolution produced no scoreable variant")
        score, name, path, sha, prof = best
        os.makedirs(winner_dir, exist_ok=True)
        shutil.copyfile(path, os.path.join(winner_dir, "work_variant.py"))
        with open(os.path.join(winner_dir, "evidence.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "variant": name,
                    "policy_sha256": sha,
                    "metric": spec.metric,
                    "sim_score": score,
                    "direction": spec.direction,
                    "source": "frontier_sim",
                    "trials": trials,
                    "doctrine": "sim scores are a SEARCH signal only",
                },
                fh,
                indent=2,
            )
        with open(
            os.path.join(winner_dir, "REQUIRES_REAL_VLLM_VERIFICATION"), "w", encoding="utf-8"
        ) as fh:
            fh.write(
                "This sim-winner is a PROPOSAL. Adoption requires a real-vLLM "
                "verified gain (ve bench --backend remote + ve verify-gain).\n"
            )
        # marker_verified comes from the BENCH profile — False under frontier_sim, so the
        # orchestrator can never adopt this candidate (search-only by construction).
        return Candidate(
            target="code:schedule_batch",
            kind="code",
            value={"policy": path},
            metrics=dict(prof.metrics) if isinstance(prof, Profile) else {},
            score=score,
            marker_verified=bool(getattr(prof, "marker_verified", False)),
            evals_used=evals,
            trials=trials,
            note=f"sim-winner: {name} ({spec.metric}={score}); search-only proposal at "
            f"{winner_dir} — adoption requires real vLLM",
        )

    return evolve_fn


def author_evolve_fn(author_fn, *, profile_fn=None):
    """Bridge a ve-author sub-agent into run_autopt's ``evolve_fn`` seam.

    The author is an UNTRUSTED PRODUCER: ``author_fn(base_config, spec)`` returns a policy source
    STRING — never a Candidate, never a marker. ``marker_verified`` is taken ONLY from the
    deterministic bench (``profile_fn``), so an author can NEVER self-promote: it has no channel to
    claim its policy ran. A marker-forging source is rejected before any bench. Off-box the default
    ``profile_fn`` is local_smoke, whose ``marker_verified`` is always False — so an authored policy
    can never adopt without a real, verified run (real-box wiring is AC7, box-gated).
    """
    import os
    import tempfile

    from vllm_evolve.bench.runtime import contains_marker_forgery

    def _bench(config: dict):
        if profile_fn is not None:
            return profile_fn(config)
        from vllm_evolve.engine.profile import collect_profile_local_smoke

        return collect_profile_local_smoke(config)

    def evolve_fn(base_config: dict, spec: Spec) -> Candidate:
        source = author_fn(base_config, spec)
        if not isinstance(source, str) or not source.strip():
            return _unverified("ve-author produced no policy source")
        if contains_marker_forgery(source):
            return _unverified("policy source emits a reserved marker (forgery) — rejected")
        fd, path = tempfile.mkstemp(suffix="_work.py", prefix="ve_author_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(source)
        prof = _bench({**base_config, "runner_kind": "candidate", "policy": path})
        # marker_verified comes from the BENCH, never the author. isinstance guards a wiring bug.
        verified = isinstance(prof, Profile) and bool(prof.marker_verified)
        if not verified:
            try:
                os.unlink(path)  # untrusted -> don't leave the policy file around
            except OSError:
                pass
            return _unverified("plugin NOT verified by the bench — untrusted")
        # value carries the policy PATH (the run_autopt config convention: config['policy'] is a
        # path), kept on disk so the orchestrator's re-bench loads the same policy.
        return Candidate(
            target="code:schedule_batch",
            kind="code",
            value={"policy": path},
            marker_verified=True,
            note="plugin verified by the bench",
        )

    return evolve_fn
