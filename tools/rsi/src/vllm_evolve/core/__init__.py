"""Frozen measurement + judgment core (L1).

Houses the deterministic data models and the accept/verify judgment that must
stay **agent-free** — no LLM / orchestration / routing imports may ever reach
this package (guarded by tests/test_frozen_core.py). Allowed dependency
direction: ``core -> bench`` only; ``bench -> core`` is forbidden.
See DESIGN.md for the dependency boundary.
"""
