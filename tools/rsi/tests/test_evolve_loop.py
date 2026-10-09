"""A: the generational engine is genuinely genetic (Evolution v2). Offline, pure.

§5.1  genetic channel: a fake author can only produce good children BECAUSE the context
      carries parent scores — and the context provably carries them.
§5.1b budget: a generation/repair storm cannot exceed max_total_evals (charged at eval_fn).
§5.2  repair: verify errors flow back; the author's fix passes; repair_count recorded.
§5.4  elitism + dedup: identical code never re-evaluated; last-gen best >= gen-0 best.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.core.schemas import Profile, Spec  # noqa: E402
from vllm_evolve.engine.evolve_loop import run_evolution  # noqa: E402

_SPEC = Spec(metric="tok_s", direction="max")
_DIAG = {"bottleneck": "scheduling_queue", "evidence_refs": ["waiting=8"]}


def _policy(quality: int) -> str:
    # a valid, verifiable policy whose "quality" is encoded in a constant the fake eval reads.
    # It clears the FULL ve-verify gate the evolution loop now reuses: skeleton arg parity + the
    # required return annotation (the real author template carries the same).
    return (f"QUALITY = {quality}\n"
            "from targets.scheduling.skeleton import ScheduleDecision\n"
            "def schedule_batch(waiting_requests, running_requests, max_num_batched_tokens,\n"
            "                   max_num_seqs, available_kv_blocks,\n"
            "                   prefix_cache_hit_rate) -> ScheduleDecision:\n"
            "    return None\n")


def _eval_by_quality(calls=None):
    def eval_fn(config):
        if calls is not None:
            calls.append(config)
        src = Path(config["policy"]).read_text(encoding="utf-8")
        q = int(src.split("QUALITY = ")[1].split("\n")[0])
        return Profile(config=config, metrics={"tok_s": float(q)}, vllm={}, gpu={},
                       marker_verified=False, source="frontier_sim",
                       outcome_class="simulator_nonqualifying")
    return eval_fn


def test_verify_reuses_the_full_ve_verify_gate():
    # the evolution loop must run an evolved variant through the SAME static gate as `ve verify`
    # (signature parity + return annotation), not just a name-exists check (Codex review P2). A
    # schedule_batch with the wrong args / missing return annotation is rejected before any eval.
    from vllm_evolve.engine.evolve_loop import _verify
    bad = "def schedule_batch(wrong_arg):\n    return []\n"
    issues = _verify(bad)
    assert issues, "a malformed schedule_batch must be rejected by the full verifier"
    assert any("L2" in i or "annotation" in i.lower() for i in issues)
    # the well-formed family template still passes the gate (real evolution is not broken)
    assert _verify(_policy(7)) == []


def test_verify_rejects_bridge_runtime_type_redefinition():
    from vllm_evolve.engine.evolve_loop import _verify

    source = (
        "class RequestInfo:\n"
        "    remaining_output_tokens = 0\n"
        + _policy(7)
    )
    issues = _verify(source)
    assert any(
        "bridge-owned runtime class" in issue and "RequestInfo" in issue
        for issue in issues
    )


def test_genetic_channel_feedback_reaches_author_and_improves_children(tmp_path):
    seen_ctx = []

    def author(ctx):
        seen_ctx.append(ctx)
        scored_parents = [p for p in ctx.parents if p.get("score") is not None]
        if not scored_parents:
            return _policy(10 + len(seen_ctx))          # blind first generation
        # the author can ONLY do this because the context carries parent scores
        best = max(p["score"] for p in scored_parents)
        return _policy(int(best) + 5)

    res = run_evolution(author, _eval_by_quality(), _SPEC, {"profile": "throughput"}, _DIAG,
                        generations=2, population=3, repair_limit=0,
                        variants_dir=str(tmp_path / "v"))
    # gen-0 children score 11/12/13. The unstructured custom crossover is rejected because it
    # cannot prove a concrete component from both parents; the valid mutation still consumes its
    # selected parent's score (12) and improves to 17.
    assert res.history[0]["best_score"] == 13.0
    assert res.history[1]["best_score"] == 17.0
    gen1_ctx = [c for c in seen_ctx if c.generation == 1]
    assert gen1_ctx and all(p["score"] is not None for c in gen1_ctx for p in c.parents)
    assert all("waiting=8" in str(c.diagnosis) for c in seen_ctx)   # diagnosis reaches author
    assert res.winner.score == 17.0 and res.winner.generation == 1


def test_nonranking_diagnostics_and_source_reach_the_next_generation(tmp_path):
    seen_ctx = []

    def author(ctx):
        seen_ctx.append(ctx)
        return _policy(10 + len(seen_ctx))

    def invalid_eval(config):
        return Profile(
            config=config,
            metrics={},
            marker_verified=True,
            source="real_vllm",
            outcome_class="eval_result",
            evidence=[
                {
                    "fitness_reasons": [
                        "workload validity is not valid_saturated_real_vllm"
                    ],
                    "diagnostics": {
                        "evidence_role": "diagnostic_only_nonranking",
                        "selection_eligible": False,
                        "raw_gain_pct": 7.25,
                        "gpu_pressure": [{"utilization_p10_pct": 38.0}],
                    },
                }
            ],
        )

    res = run_evolution(
        author,
        invalid_eval,
        _SPEC,
        {},
        _DIAG,
        generations=2,
        population=3,
        repair_limit=0,
        max_total_evals=6,
        variants_dir=str(tmp_path / "v"),
    )
    gen1 = next(ctx for ctx in seen_ctx if ctx.generation == 1)
    evidence = [
        row
        for row in gen1.lessons
        if row.get("kind") == "prior_candidate_evidence"
    ]
    assert evidence
    assert evidence[0]["selection_eligible"] is False
    assert evidence[0]["diagnostics"]["raw_gain_pct"] == 7.25
    assert "QUALITY =" in evidence[0]["source"]
    assert res.winner is None


def test_budget_is_pinned_at_eval_boundary(tmp_path):
    calls = []
    n = [0]

    def author(ctx):
        n[0] += 1
        return _policy(n[0])          # every child unique -> every child wants an eval

    res = run_evolution(author, _eval_by_quality(calls), _SPEC, {}, _DIAG,
                        generations=10, population=3, repair_limit=1,
                        max_total_evals=5, variants_dir=str(tmp_path / "v"))
    assert len(calls) == 5 and res.evals_used == 5
    assert res.terminated == "budget_exhausted"


def test_repair_error_feedback_fixes_the_child(tmp_path):
    def author(ctx):
        if not ctx.last_errors:
            return "def wrong_name():\n    return None\n"   # fails L2 (missing schedule_batch)
        assert any("schedule_batch" in e for e in ctx.last_errors)   # real error text arrived
        return _policy(30)

    res = run_evolution(author, _eval_by_quality(), _SPEC, {}, _DIAG,
                        generations=1, population=1, repair_limit=1,
                        variants_dir=str(tmp_path / "v"))
    assert res.winner is not None and res.winner.score == 30.0
    assert res.winner.verify_ok is True and res.winner.repair_count == 1


def test_dedup_and_elitism(tmp_path):
    calls = []

    def author(ctx):
        return _policy(42)            # every child identical -> one eval total

    res = run_evolution(author, _eval_by_quality(calls), _SPEC, {}, _DIAG,
                        generations=4, population=3, repair_limit=0,
                        variants_dir=str(tmp_path / "v"))
    assert len(calls) == 1            # identical sha evaluated exactly once
    assert res.history[-1]["best_score"] >= res.history[0]["best_score"]   # elitism monotone
    # gen0 improves (first score), gen1/gen2 flat -> 2 consecutive no-improve -> honest early stop
    assert res.terminated == "no_improvement" and res.generations_completed == 3


def test_orchestrate_maps_evolution_to_candidate_and_never_adopts(tmp_path):
    # F: run_autopt + author_fn drives the generational engine; the winner maps back to a
    # Candidate (trajectory keys preserved) and — marker_verified False under sim — the
    # adoption guard records no_verified_candidate. Locks intact.
    from vllm_evolve.engine.orchestrate import run_autopt

    def author(ctx):
        return _policy(10 + len(ctx.peers))

    def eval_fn(config):
        # scheduling_queue signature so code:schedule_batch ranks in; policy evals score by file
        if "policy" in config and Path(str(config.get("policy", ""))).exists():
            return _eval_by_quality()(config)
        return Profile(config=config, metrics={"tok_s": 5.0},
                       vllm={"kv_util": 0.2, "waiting": 8, "preempt": 0},
                       gpu={"duty_cycle": 0.7}, marker_verified=False,
                       source="frontier_sim", outcome_class="simulator_nonqualifying")

    res = run_autopt(_SPEC, eval_fn=eval_fn, author_fn=author,
                     evolve_params={"generations": 2, "population": 2,
                                    "variants_dir": str(tmp_path / "v")},
                     base_config={"profile": "throughput"}, max_rounds=2)
    code_rounds = [r for r in res["rounds"]
                   if r.get("searched_target") == "code:schedule_batch"]
    assert code_rounds, res["rounds"]
    rec = code_rounds[0]
    cand = rec["candidate"]
    assert cand["value"].get("policy") and cand["score"] is not None      # trajectory keys live
    assert cand["marker_verified"] is False and cand["trials"]
    assert rec["evolution"]["generations_completed"] >= 1
    assert rec["verdict"] == {"verdict": "no_verified_candidate"}          # never adopts on sim
    assert res["adopted"] is None and res["outcome"] == "dod_b"


def test_lessons_written_only_at_run_end(tmp_path):
    from vllm_evolve.store.db import Store
    writes_during_run = []

    class SpyStore(Store):
        def put_lesson(self, **kw):
            writes_during_run.append(kw)
            return super().put_lesson(**kw)

    n = [0]

    def author(ctx):
        n[0] += 1
        assert not writes_during_run     # nothing persisted while authoring is still happening
        return _policy(n[0])

    with SpyStore(tmp_path / "s.db") as store:
        res = run_evolution(author, _eval_by_quality(), _SPEC, {"profile": "throughput"},
                            _DIAG, generations=2, population=2, repair_limit=0,
                            store=store, run_id="t1", variants_dir=str(tmp_path / "v"))
        assert res.lessons_written and len(writes_during_run) == len(res.lessons_written)
        got = store.get_lessons(regime="throughput", limit=10)
        assert any("run summary" in r["conclusion"] for r in got)
        assert all(r["source"] == "frontier_sim" for r in got)
