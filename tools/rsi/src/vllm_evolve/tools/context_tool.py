"""
Context tool — assemble evolution context from the store.

Queries the DB for current state (best policy, recent evaluations,
failures, lineage) and formats it for the scaffold.
"""
from __future__ import annotations

import json


def build_context(
    target_name: str = "scheduling",
    parent_id: str | None = None,
    n_inspirations: int = 3,
    n_failures: int = 5,
) -> dict:
    """Build structured context for a scaffold.

    Returns a dict with all the information an agent needs to make
    a good mutation decision. Can be serialized to JSON or formatted
    as flat text.

    Args:
        target_name: Which target to get context for
        parent_id: Specific parent to mutate (None = use best)
        n_inspirations: How many high-fitness examples to include
        n_failures: How many recent failures to include
    """
    from vllm_evolve.tools.store_tool import get_store

    store = get_store()

    # Get parent (explicit or best)
    parent = None
    if parent_id:
        parent = store.get_policy(parent_id)
    if parent is None:
        best_list = store.best_policies(target_name, n=1)
        if best_list:
            parent = store.get_policy(best_list[0]["policy_id"])

    # Get inspirations (top-N excluding parent)
    inspirations = []
    top_policies = store.best_policies(target_name, n=n_inspirations + 1)
    for p in top_policies:
        if parent and p["policy_id"] == parent.get("policy_id"):
            continue
        pol = store.get_policy(p["policy_id"])
        if pol:
            inspirations.append({
                "policy_id": p["policy_id"],
                "fitness": p.get("avg_fitness"),
                "generation": p.get("generation"),
                "code": pol["source_code"],
            })
        if len(inspirations) >= n_inspirations:
            break

    # Get parent's evaluation history
    parent_evals = []
    if parent:
        parent_evals = store.history("policy", parent["policy_id"])

    # Get recent failed checks
    failures = []
    try:
        failures = store.query(
            "SELECT * FROM checks WHERE item_type='policy' AND passed=0 "
            "ORDER BY created_at DESC LIMIT ?",
            (n_failures,),
        )
    except Exception:
        pass  # store unavailable

    # Get seed metrics (generation 0)
    seed_metrics = {}
    try:
        rows = store.query(
            "SELECT e.metrics_json FROM evaluations e "
            "JOIN lineage l ON e.item_id = l.policy_id "
            "WHERE l.generation = 0 AND e.item_type = 'policy' "
            "LIMIT 1"
        )
        if rows:
            seed_metrics = json.loads(rows[0]["metrics_json"])
    except Exception:
        pass  # store unavailable

    # Load target spec for metrics and domain hints
    from pathlib import Path
    metrics_info = []
    evolvable_functions = []
    domain_hints = ""
    try:
        from vllm_evolve.config import load_target_spec
        spec = load_target_spec(target_name)
        metrics_info = [
            {"name": m.name, "direction": m.direction, "weight": m.weight}
            for m in spec.metrics
        ]
        evolvable_functions = [
            {"name": fn["name"], "signature": fn.get("signature", "")}
            for fn in spec.evolvable_functions
        ]
        hints_path = Path(getattr(spec, "prompt_hints_path", ""))
        if hints_path.exists():
            domain_hints = hints_path.read_text(encoding="utf-8")
    except Exception:
        pass

    # Build context dict
    ctx = {
        "target_name": target_name,
        "parent": {
            "policy_id": parent["policy_id"] if parent else None,
            "code": parent["source_code"] if parent else "",
            "evaluations": parent_evals,
        } if parent else None,
        "inspirations": inspirations,
        "seed_metrics": seed_metrics,
        "metrics_info": metrics_info,
        "evolvable_functions": evolvable_functions,
        "domain_hints": domain_hints,
        "recent_failures": failures,
        "store_stats": store.stats(),
    }

    return ctx


def format_flat(ctx: dict) -> str:
    """Format context as flat text for scaffolds without tool_use."""
    sections = []

    sections.append(f"=== CONTEXT: {ctx['target_name']} ===")

    if ctx.get("parent"):
        p = ctx["parent"]
        sections.append(f"\n── PARENT ({p['policy_id']}) ──")
        sections.append(f"```python\n{p['code']}\n```")
        if p.get("evaluations"):
            latest = p["evaluations"][-1]
            sections.append(f"Latest fitness: {latest.get('fitness', '?')}")

    if ctx.get("metrics_info"):
        sections.append("\n── METRICS (direction + weight) ──")
        for m in ctx["metrics_info"]:
            sections.append(f"  {m['name']}: {m['direction']} (weight={m['weight']})")

    if ctx.get("evolvable_functions"):
        sections.append("\n── EVOLVABLE FUNCTIONS ──")
        for fn in ctx["evolvable_functions"]:
            sections.append(f"  {fn['name']}: {fn['signature']}")

    if ctx.get("seed_metrics"):
        sections.append("\n── SEED METRICS ──")
        for k, v in ctx["seed_metrics"].items():
            sections.append(f"  {k}: {v}")

    if ctx.get("domain_hints"):
        sections.append("\n── DOMAIN HINTS ──")
        # Include first 2000 chars to keep prompt size manageable
        sections.append(ctx["domain_hints"][:2000])

    for i, insp in enumerate(ctx.get("inspirations", [])):
        sections.append(
            f"\n── INSPIRATION {chr(65+i)} (fitness={insp['fitness']}) ──"
        )
        sections.append(f"```python\n{insp['code']}\n```")

    if ctx.get("recent_failures"):
        sections.append("\n── RECENT FAILURES ──")
        for f in ctx["recent_failures"]:
            issues = json.loads(f.get("issues_json", "[]"))
            sections.append(f"  - {'; '.join(issues)}")

    stats = ctx.get("store_stats", {})
    sections.append(f"\n── STORE: {stats.get('policies', 0)} policies, "
                    f"{stats.get('evaluations', 0)} evaluations ──")

    return "\n".join(sections)


# --- Round 1 (Codex review): route public callables through phase guard ---
from vllm_evolve.tools._legacy_guard import _install_guard as _install_legacy_guard  # noqa: E402

_install_legacy_guard(__name__, ("build_context", "format_flat",))
