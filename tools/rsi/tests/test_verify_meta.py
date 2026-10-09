"""Meta-tests over ``tests/fixtures/bad_policies/``.

These tests prevent silent drift between the harness's published catalog of
safety categories (``vllm_evolve.tools._safety_catalog``) and the set of
fixture files that exercise each rejection path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.tools._safety_catalog import CATEGORIES  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "bad_policies"


def _fixture_names() -> set[str]:
    """Return the set of fixture category names by file stem.

    ``__init__.py`` is excluded so the test does not see a sentinel category.
    """
    return {
        p.stem
        for p in FIXTURE_DIR.glob("*.py")
        if p.is_file() and p.stem != "__init__"
    }


def test_fixture_list_matches_safety_catalog() -> None:
    fixtures = _fixture_names()
    missing_fixtures = CATEGORIES - fixtures
    orphan_fixtures = fixtures - CATEGORIES
    assert not missing_fixtures, (
        f"Safety catalog declares categories with no fixture: "
        f"{sorted(missing_fixtures)}"
    )
    assert not orphan_fixtures, (
        f"Fixture directory has files outside the safety catalog: "
        f"{sorted(orphan_fixtures)}"
    )


@pytest.mark.parametrize("category", sorted(CATEGORIES))
def test_each_fixture_file_exists(category: str) -> None:
    path = FIXTURE_DIR / f"{category}.py"
    assert path.exists(), f"Missing fixture for safety category {category!r}: {path}"
    assert path.stat().st_size > 0, f"Empty fixture file: {path}"
