"""
LLM diff parser and applicator.

Supports the SEARCH/REPLACE block format used by the evolutionary operators:

    <<<SEARCH
    <original code>
    >>>
    <<<REPLACE
    <replacement code>
    >>>

Multiple hunks may appear in a single LLM output, separated by any amount of
whitespace / prose.  The parser is intentionally lenient about surrounding
text so that the LLM can include explanations between blocks.

Public API
----------
- ``parse_search_replace(llm_output)`` → list of (search, replace) tuples
- ``apply_diff(source, hunks)``        → patched source string or None
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Markers (accept optional trailing whitespace after the literal)
_SEARCH_OPEN = r"<<<\s*SEARCH\s*\n"
_SEARCH_CLOSE = r"\s*>>>\s*\n?"
_REPLACE_OPEN = r"<<<\s*REPLACE\s*\n"
_REPLACE_CLOSE = r"\s*>>>\s*"

# Full hunk pattern (non-greedy content capture)
_HUNK_RE = re.compile(
    _SEARCH_OPEN
    + r"(.*?)"  # group 1: search text
    + _SEARCH_CLOSE
    + _REPLACE_OPEN
    + r"(.*?)"  # group 2: replace text
    + _REPLACE_CLOSE,
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def parse_search_replace(llm_output: str) -> list[tuple[str, str]]:
    """Parse SEARCH/REPLACE hunks from *llm_output*.

    The parser handles:
    - Leading/trailing prose around blocks (ignored).
    - Multiple hunks in a single output.
    - Optional whitespace after ``<<<SEARCH`` / ``<<<REPLACE`` / ``>>>``
      markers.
    - Windows-style ``\\r\\n`` line endings (normalised to ``\\n`` before
      parsing).

    Args:
        llm_output: Raw text returned by the LLM, which may contain one or
                    more SEARCH/REPLACE blocks plus free-form explanation.

    Returns:
        A list of ``(search, replace)`` string tuples, one per hunk, in the
        order they appear in *llm_output*.  Returns an empty list if no valid
        hunk is found.
    """
    if not llm_output:
        return []

    # Normalise line endings
    text = llm_output.replace("\r\n", "\n").replace("\r", "\n")

    hunks: list[tuple[str, str]] = []
    for match in _HUNK_RE.finditer(text):
        search_text = match.group(1)
        replace_text = match.group(2)
        # Strip exactly one leading newline that the marker swallowed, and
        # one trailing newline before the closing >>>. The regex already
        # consumed the newline after the opening marker; we may still have a
        # trailing newline inside the captured group from a generous ``.*?``.
        # Preserve internal whitespace faithfully.
        search_text = _strip_outer_newline(search_text)
        replace_text = _strip_outer_newline(replace_text)
        hunks.append((search_text, replace_text))

    return hunks


def apply_diff(source: str, hunks: list[tuple[str, str]]) -> str | None:
    """Apply a sequence of SEARCH/REPLACE hunks to *source*.

    Hunks are applied in order.  Each hunk's search string must appear
    *exactly* (case-sensitively) in the current state of the source after
    all previous hunks have been applied.

    Args:
        source: The original source code string.
        hunks:  A list of ``(search, replace)`` tuples as returned by
                :func:`parse_search_replace`.

    Returns:
        The patched source string on success, or ``None`` if any hunk's
        search string cannot be found in the (possibly already partially
        patched) source.  When ``None`` is returned the caller should treat
        the entire diff as failed; no partial application is exposed.
    """
    if not hunks:
        return source

    current = source
    for i, (search, replace) in enumerate(hunks):
        if search in current:
            # Exact match — best case
            current = current.replace(search, replace, 1)
        else:
            # Fuzzy fallback: try with normalized whitespace
            # This catches trailing spaces, tab/space differences, etc.
            search_stripped = search.rstrip()
            if search_stripped in current:
                # Search had trailing whitespace — strip and retry
                current = current.replace(search_stripped, replace, 1)
                continue
            normalized_search = _normalize_ws(search)
            matched = False
            for line_start in range(len(current)):
                # Find a region in current that matches after normalization
                candidate = current[line_start:line_start + len(search) + 50]
                if _normalize_ws(candidate[:len(search)]) == normalized_search:
                    # Found fuzzy match — use exact bounds
                    # Try to find the exact end by matching line count
                    search_lines = search.count('\n')
                    pos = line_start
                    for _ in range(search_lines + 1):
                        nl = current.find('\n', pos)
                        if nl == -1:
                            pos = len(current)
                            break
                        pos = nl + 1
                    actual_search = current[line_start:pos].rstrip('\n')
                    if _normalize_ws(actual_search) == normalized_search:
                        current = current[:line_start] + replace + current[pos:]
                        matched = True
                        break
                if line_start > 5000:  # don't scan forever
                    break
            if not matched:
                return None

    return current


def _normalize_ws(text: str) -> str:
    """Normalize whitespace for fuzzy matching: strip trailing, collapse runs."""
    lines = text.split('\n')
    return '\n'.join(line.rstrip() for line in lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_outer_newline(text: str) -> str:
    """Remove at most one leading and one trailing newline from *text*.

    This compensates for the fact that the regex markers consume the newline
    immediately after the ``<<<`` / ``>>>`` delimiters, leaving the captured
    content with a trailing ``\\n`` from the last line of the block.
    """
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    return text
