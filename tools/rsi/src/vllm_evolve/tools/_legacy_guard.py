"""Round-1 fix per Codex's Round-0 review: prevent direct submodule import
from bypassing the phase guard.

Emptying ``tools/__init__.py`` alone only blocks
``from vllm_evolve.tools import simulate`` (package attribute lookup). It
does **not** block
``from vllm_evolve.tools.simulate import simulate`` because Python imports
the submodule directly and reads the callable out of its globals.

Each legacy tool module calls :func:`_install_guard` at module bottom with
the list of public callables it wishes to protect. The helper rebinds
those names so any access -- whether by submodule import, by
``getattr``, or by attribute lookup on the already-imported module --
routes through ``phase_guard.check_or_fail("legacy_tool")`` before
delegating to the original callable.

Round 1 registers ``legacy_tool`` as an administrative verb (no
required phase) so the existing ``vllm-evolve`` CLI workflows keep
working while the hook point is present. Future rounds can tighten by
re-registering the verb with a more restrictive required phase once
``vllm-evolve`` is removed in M5 (DEC-6).
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from functools import wraps


def _install_guard(module_name: str, public_names: Iterable[str]) -> None:
    """Wrap each name in ``public_names`` on the module so calls route
    through ``phase_guard.check_or_fail('legacy_tool')`` before the
    real implementation runs.

    The wrapping is idempotent: re-importing the module is safe because
    each wrapper records the original on ``wrapper._unguarded``.
    """

    mod = sys.modules[module_name]
    for name in public_names:
        existing = getattr(mod, name, None)
        if existing is None or not callable(existing):
            continue
        if getattr(existing, "_legacy_guard_wrapped", False):
            # Module already guarded (e.g. test reimport). Skip.
            continue

        real = existing  # captured into closure via default arg below

        @wraps(real)
        def wrapper(*args, _real=real, _verb="legacy_tool", **kwargs):
            from vllm_evolve.tools import phase_guard as _pg

            _pg.check_or_fail(_verb)
            return _real(*args, **kwargs)

        wrapper._legacy_guard_wrapped = True
        wrapper._unguarded = real
        setattr(mod, name, wrapper)


__all__ = ["_install_guard"]
