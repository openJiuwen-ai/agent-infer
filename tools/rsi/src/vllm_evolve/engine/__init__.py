"""Evolution engine (L2): the untrusted orchestration / search pipeline.

The auto (``ve autopt``) flow lives here — profile → diagnose → target-select →
optimize / evolve → orchestrate, plus the L4 verify driver wiring. This layer MAY
import ``core/`` (frozen judgment) and ``bench/`` (measurement); the frozen core
MUST NOT import ``engine/`` (guarded by tests/test_frozen_core.py).
See DESIGN.md for the dependency boundary.
"""
