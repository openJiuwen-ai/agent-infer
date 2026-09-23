"""Top-level usage modes (L2 orchestration): ``autopt`` / ``tune`` / ``port``.

Each mode is a composite pipeline that drives the existing gated steps — it never
bypasses the phase machine or the adoption gate. A mode's result reaches a
gain/keep only through the frozen ``compare`` / ``verify-gain`` / ``accept`` path
(``bench.eval_result.real_source_block``), so a synthetic ``local_smoke`` run can
never become a gain/keep/AC6. ``research`` is NOT a mode — it is a step inside the
``autopt`` flow (``ve-research`` + ``knowledge/``). See DESIGN.md for the mode boundaries.

This layer is UNTRUSTED and MUST NOT appear in the frozen measure/judge closure
(``tests/test_frozen_core.py`` forbids ``modes/``).
"""
