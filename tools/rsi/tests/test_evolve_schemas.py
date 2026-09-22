"""B: AuthorContext.to_prompt() is the single full-field serialization source (Evolution v2).

The snapshot pins EVERY dataclass field name + the doctrine line in the rendering, so neither
author form (template / ve-author agent) can silently drop the feedback that makes evolution
genetic. Pure, offline.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.engine.evolve_schemas import (  # noqa: E402
    DOCTRINE,
    AuthorContext,
    EvolutionResult,
    Lesson,
    LineageEntry,
    prompt_source_author,
)


def test_to_prompt_covers_every_field_and_doctrine():
    ctx = AuthorContext(
        spec={"metric": "goodput_req_s", "direction": "max"},
        diagnosis={"bottleneck": "scheduling_queue", "evidence_refs": ["waiting=8"]},
        skeleton="def schedule_batch(...): ...",
        parents=[{"sha": "p1", "score": 17.3, "generation": 0, "source": "src"}],
        peers=[{"sha": "q1", "score": 18.0}],
        lessons=[{"lesson_id": 7, "conclusion": "tok_s is work-conserving"}],
        last_errors=["L2: signature mismatch"],
        budget={"evals_remaining": 9, "repair_remaining": 1},
        generation=2,
    )
    prompt = ctx.to_prompt()
    for f in fields(AuthorContext):           # EVERY field name appears — none can be dropped
        assert f'"{f.name}"' in prompt, f"to_prompt dropped field {f.name!r}"
    assert DOCTRINE in prompt                  # the doctrine line is present verbatim
    # representative values made it through (parent score + failure reason + lesson)
    assert "17.3" in prompt and "L2: signature mismatch" in prompt
    assert "tok_s is work-conserving" in prompt
    # deterministic: same context -> identical rendering
    assert prompt == ctx.to_prompt()
    json.loads(prompt)                         # structured JSON, parseable


def test_lineage_lesson_result_roundtrip():
    e = LineageEntry(sha="abc", generation=1, parent_sha="p", score=18.0,
                     source_path="x.py", verify_ok=True)
    assert e.to_dict()["parent_sha"] == "p"
    lesson = Lesson(run_id="r1", source="frontier_sim", policy_sha="abc",
                    regime="throughput", metric="goodput_req_s", score=18.0,
                    conclusion="SJF beats LJF under TTFT SLO")
    assert lesson.to_dict()["source"] == "frontier_sim"
    res = EvolutionResult(winner=e, archive=[e], generations_completed=2, evals_used=7,
                          terminated="no_improvement",
                          history=[{"generation": 0, "best_sha": "abc", "best_score": 18.0}])
    d = res.to_dict()
    assert d["winner"]["sha"] == "abc" and d["terminated"] == "no_improvement"


def test_prompt_source_author_passes_exact_context_and_returns_source():
    ctx = AuthorContext(
        diagnosis={"bottleneck": "queue"},
        parents=[{"sha": "parent", "score": 3.0, "source": "parent source"}],
        lessons=[{"conclusion": "use bounded aging"}],
        last_errors=["L2 failure"],
        budget={"evals_remaining": 4},
    )
    seen = []

    def complete(prompt: str) -> str:
        seen.append(prompt)
        return "def schedule_batch():\n    pass\n"

    author = prompt_source_author(complete)
    assert author(ctx).startswith("def schedule_batch")
    assert seen == [ctx.to_prompt()]
