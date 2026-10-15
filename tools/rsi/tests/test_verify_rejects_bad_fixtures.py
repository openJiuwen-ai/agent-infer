"""End-to-end test of ``ve verify`` over ``tests/fixtures/bad_policies/``.

Each bad fixture is asserted to be rejected with exit code 2 and an output
JSON whose ``outcome_class`` is ``safety_rejection``. The matching positive
case uses ``targets/scheduling/seed.py``, which is the project's FCFS
baseline and should always pass L1 + L2 verification.

State is isolated by monkey-patching ``phase_guard._repo_root`` so tests do
not stomp on the real ``.ar/state/`` directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.cli import main as cli_main  # noqa: E402
from vllm_evolve.tools import phase_guard  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "bad_policies"
SEED_POLICY = REPO_ROOT / "targets" / "scheduling" / "seed.py"


def _walk_to_generate() -> None:
    """Walk the transition table from INIT to GENERATE inside the temp root."""
    phase_guard.reset_state()
    phase_guard.transition(phase_guard.Phase.READ_CONTEXT)
    phase_guard.transition(phase_guard.Phase.DESIGN)
    phase_guard.transition(phase_guard.Phase.GENERATE)


@pytest.fixture(autouse=True)
def _isolated_phase_state(tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    monkeypatch.setattr(phase_guard, "_repo_root", lambda: fake_root)
    _walk_to_generate()
    yield
    phase_guard.reset_state()


def _run(argv: list[str], capsys) -> tuple[int, dict]:
    rc = cli_main.main(argv)
    out = capsys.readouterr().out.strip()
    assert out, f"ar {argv!r} produced no stdout"
    payload = json.loads(out)
    return rc, payload


_BAD_FIXTURES = sorted(
    p.stem
    for p in FIXTURE_DIR.glob("*.py")
    if p.is_file() and p.stem != "__init__"
)


@pytest.mark.parametrize("category", _BAD_FIXTURES)
def test_bad_fixture_is_rejected(category, capsys):
    fixture = FIXTURE_DIR / f"{category}.py"
    rc, payload = _run(["verify", str(fixture), "scheduling"], capsys)
    assert rc == 2, f"{category}: expected exit 2, got {rc}: {payload}"
    assert payload["ok"] is False, payload
    assert payload["outcome_class"] == "safety_rejection", payload
    assert payload["issues"], f"{category}: empty issues list: {payload}"


def test_clean_seed_policy_passes_verify(capsys):
    assert SEED_POLICY.exists(), f"Seed policy missing at {SEED_POLICY}"
    rc, payload = _run(["verify", str(SEED_POLICY), "scheduling"], capsys)
    assert rc == 0, f"seed policy expected to pass: {payload}"
    assert payload["ok"] is True
    assert payload["outcome_class"] == "verify_pass"
    # ve verify on success must record VERIFY_PASSED in the phase state.
    assert phase_guard.read_phase() == phase_guard.Phase.VERIFY_PASSED


def test_missing_policy_file(capsys, tmp_path):
    rc, payload = _run(
        ["verify", str(tmp_path / "does-not-exist.py"), "scheduling"], capsys
    )
    assert rc == 2
    assert payload["outcome_class"] == "policy_not_found"
