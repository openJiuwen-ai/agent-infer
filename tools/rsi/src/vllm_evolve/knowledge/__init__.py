"""Knowledge layer for the ``ve-research`` step of the auto (autopt) flow.

Two evidence planes remain deliberately separate:

* ``gather`` reads immutable *internal experiment memory* from SQLite.
* ``ResearchCompiler`` builds a frozen *external/domain expert brief* from
  reviewed and optional live primary sources.

Both are untrusted proposal inputs.  Neither module can write a benchmark score
or an adoption verdict.

Dependency direction: ``knowledge -> store`` only. The frozen measure/judge core
(``bench/`` + ``core/``) MUST NOT import ``knowledge`` (forbidden by
``tests/test_frozen_core.py``).
"""
from vllm_evolve.knowledge.compiler import (
    ResearchCompiler,
    live_required_compiler,
    load_research_context,
    materialize_research_context,
    offline_compiler,
)
from vllm_evolve.knowledge.research import gather
from vllm_evolve.knowledge.schemas import (
    CandidateManifest,
    MechanismCard,
    ResearchContext,
    ResearchSource,
)

__all__ = [
    "CandidateManifest",
    "MechanismCard",
    "ResearchCompiler",
    "ResearchContext",
    "ResearchSource",
    "gather",
    "load_research_context",
    "live_required_compiler",
    "materialize_research_context",
    "offline_compiler",
]
