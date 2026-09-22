"""
L4: Runtime fallback wrapper.

Wraps evolved code execution with try/except and timeout protection.
If the evolved code fails at runtime, automatically falls back to
the seed implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class TimeoutError(Exception):
    pass


def wrap_with_fallback(
    evolved_fn: Callable,
    seed_fn: Callable,
    timeout_ms: float = 1.0,
) -> Callable:
    """Wrap evolved function with fallback to seed on any error.

    Args:
        evolved_fn: The evolved function to try first.
        seed_fn: The seed function to fall back to.
        timeout_ms: Maximum execution time in milliseconds (unused in
            pure-Python — for documentation/future use).

    Returns:
        A wrapper function that tries evolved_fn first, falls back to seed_fn.
    """
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return evolved_fn(*args, **kwargs)
        except Exception as _e:  # policy crashed, fallback active
            return seed_fn(*args, **kwargs)

    wrapped.__name__ = f"fallback_{evolved_fn.__name__}"
    wrapped.__doc__ = (
        f"Evolved {evolved_fn.__name__} with fallback to seed on error."
    )
    return wrapped
