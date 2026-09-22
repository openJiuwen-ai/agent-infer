"""C: the headless template author consumes feedback and stays within the contract. Offline."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench.runtime import contains_marker_forgery  # noqa: E402
from vllm_evolve.engine.evolve_schemas import AuthorContext  # noqa: E402
from vllm_evolve.engine.evolve_target import template_author_fn  # noqa: E402
from vllm_evolve.trust.safety import check_safety, check_signatures  # noqa: E402


def _ctx(**kw) -> AuthorContext:
    return AuthorContext(spec={"metric": "tok_s", "direction": "max"}, **kw)


def test_gen0_seeds_distinct_valid_families():
    sources = {template_author_fn(_ctx(peers=[{"sha": str(j)} for j in range(i)]))
               for i in range(4)}
    assert len(sources) == 4                       # 4 slots -> 4 distinct family variants
    for src in sources:
        assert check_safety(src).ok and check_signatures(src, ["schedule_batch"]).ok
        assert not contains_marker_forgery(src)


def test_later_generations_mutate_around_the_best_parent():
    # the best parent carries a VE_STRUCT recipe; later children mutate ONE structural dimension
    # (order/gate/switch) around it -> genuine STRUCTURAL neighbours, all valid schedulers.
    from vllm_evolve.engine.evolve_target import _build_policy
    parents = [{"sha": "a", "score": 10.0, "source": _build_policy("CF", order="cache_first")},
               {"sha": "b", "score": 99.0, "source": _build_policy("SJF", order="sjf")}]
    srcs = [template_author_fn(_ctx(parents=parents,
                                    peers=[{"sha": str(j)} for j in range(i)]))
            for i in range(4)]
    assert len(set(srcs)) >= 3                      # mutating distinct dimensions gives diversity
    for src in srcs:
        assert check_safety(src).ok and check_signatures(src, ["schedule_batch"]).ok
    assert any("VE_STRUCT:order=sjf;gate=none;switch=0" not in s for s in srcs)  # not parent clones


def test_repair_falls_back_to_seed_ordering():
    src = template_author_fn(_ctx(last_errors=["L2: Missing function: schedule_batch"]))
    assert "SJF-REPAIR" in src                     # repair -> the always-valid SJF seed
    assert check_safety(src).ok and check_signatures(src, ["schedule_batch"]).ok


def test_author_returns_source_never_touches_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                    # any file write would land here
    out = template_author_fn(_ctx())
    assert isinstance(out, str) and "def schedule_batch" in out
    assert list(tmp_path.iterdir()) == []          # contract: returns a string, writes nothing
