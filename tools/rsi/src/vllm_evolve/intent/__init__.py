"""Intent layer (L3 input): NL goal parsing (spec) + routing.

``spec.py`` is the transparent keyword/regex goal→Spec parser (no LLM). The
untrusted ``router.py`` (mode/skill selection) is added in P4. ``intent/*`` may
import ``core.schemas`` but the frozen core MUST NOT import ``intent/*`` (PLAN §1).
"""
