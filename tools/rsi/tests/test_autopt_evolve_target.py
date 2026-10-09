"""default_evolve_fn is an honest no-op: code-surface evolution is not yet wired to the real
backend (the legacy GA engine was removed), so it always returns an UNVERIFIED candidate the
orchestrator will never adopt (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.core.schemas import Spec  # noqa: E402
from vllm_evolve.engine.evolve_target import default_evolve_fn  # noqa: E402


def test_returns_unverified_candidate():
    # never a fabricated "evolved" win: empty value, not marker-verified, with an honest note.
    cand = default_evolve_fn({}, Spec(metric="goodput_req_s", direction="max"))
    assert cand.target == "code:schedule_batch" and cand.kind == "code"
    assert cand.marker_verified is False and cand.value == {}
    assert "not yet wired" in cand.note


def test_no_engine_import_dependency():
    # the rewrite must NOT import the deleted legacy engine/scaffold to produce a candidate.
    import inspect
    src = inspect.getsource(default_evolve_fn)
    assert "import" not in src   # pure no-op: needs no imports at all
