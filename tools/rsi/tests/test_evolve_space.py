"""旋钮 1: the evolution author's mutation space is genuinely OPEN — it produces STRUCTURALLY
distinct schedulers (cache-aware admission, KV back-pressure, anti-starvation, load switching), not
five textbook sort orderings — and every generated policy clears the SAME full L1/L2 verify as a
hand-authored one. Pure / offline (no Frontier, no GPU).
"""
from __future__ import annotations

from vllm_evolve.cli.main import _verify_code
from vllm_evolve.engine.evolve_schemas import AuthorContext
from vllm_evolve.engine.evolve_target import _CATALOG, _build_policy, template_author_fn


class _Ctx:
    def __init__(self, peers, parents, spec=None, last_errors=None):
        self.peers, self.parents, self.spec, self.last_errors = peers, parents, spec, last_errors

    def to_prompt(self):
        return AuthorContext(
            spec=self.spec or {},
            diagnosis={"bottleneck": "scheduling_queue"},
            skeleton="def schedule_batch(...): ...",
            parents=self.parents,
            peers=self.peers,
            lessons=[],
            last_errors=self.last_errors or [],
            budget={"evals_remaining": 10, "repair_remaining": 1},
            generation=0,
        ).to_prompt()


def _catalog_sources():
    return [_build_policy(n.upper(), **kw) for n, kw in _CATALOG]


def test_every_generated_policy_clears_the_full_verify():
    # the full ve-verify gate (L1 safety + return annotation, L2 signature parity + nested-loop +
    # fabricated-id) — a structural variant must pass exactly like a hand-authored policy.
    for src in _catalog_sources():
        assert _verify_code(src, "scheduling") == [], src.splitlines()[0]


def test_space_is_not_sort_only_and_uses_real_primitives():
    srcs = _catalog_sources()
    blob = "\n".join(srcs)
    prims = [p for p in ("prefix_cached_tokens", "num_preemptions", "available_kv_blocks",
                         "prefix_cache_hit_rate", "len(waiting_requests)") if p in blob]
    assert len(prims) >= 3, prims                       # >= 3 real primitives beyond remaining/arr
    structural = [s for s in srcs if "kv_reserve = 4" in s or "int(seq_budget" in s
                  or "if len(waiting_requests)" in s or "queue_pressure =" in s]
    assert len(structural) >= 3                          # >= 3 variants add STRUCTURE beyond a sort


def test_gen0_seeds_structural_variants():
    gen0 = [template_author_fn(_Ctx(peers=list(range(i)), parents=[]))
            for i in range(len(_CATALOG))]
    assert all(_verify_code(s, "scheduling") == [] for s in gen0)
    assert any("prefix_cached_tokens" in s or "num_preemptions" in s for s in gen0)


def test_feedback_mutates_the_parents_structure():
    # a scored parent's VE_STRUCT recipe is read and ONE dimension mutated -> a structural
    # neighbour, not a clone (feedback shapes STRUCTURE, not just a scalar).
    parent = {"score": 1.0, "source": _build_policy("SJF", order="sjf")}
    child = template_author_fn(_Ctx(peers=[1], parents=[parent], spec={"direction": "max"}))
    assert _verify_code(child, "scheduling") == []
    assert "VE_STRUCT:order=sjf;gate=none;switch=0" not in child


def test_repair_falls_back_to_a_valid_seed():
    src = template_author_fn(_Ctx(peers=[], parents=[], last_errors=["L2: boom"]))
    assert _verify_code(src, "scheduling") == []


def test_template_author_reads_the_single_to_prompt_contract():
    class PromptOnly:
        calls = 0

        def to_prompt(self):
            self.calls += 1
            return AuthorContext(
                spec={"direction": "max"},
                diagnosis={"bottleneck": "scheduling_queue"},
                skeleton="def schedule_batch(...): ...",
                parents=[],
                peers=[],
                lessons=[{"conclusion": "avoid starvation"}],
                last_errors=[],
                budget={"evals_remaining": 1, "repair_remaining": 0},
                generation=0,
            ).to_prompt()

    ctx = PromptOnly()
    assert _verify_code(template_author_fn(ctx), "scheduling") == []
    assert ctx.calls == 1
