"""AC6: the shipped schema enums cannot silently drift from the code enums / shipped regimes.

This is the guard that keeps `outcome_class`, `source`, and `regime` honest across the
schema, the `OutcomeClass` taxonomy, and the actual profile YAMLs — including the new
quarantined `local_smoke` / `local_smoke_nonqualifying` members (zero GPU).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.bench.eval_result import load_schema  # noqa: E402
from vllm_evolve.bench.outcome import (  # noqa: E402
    FAILURE_CLASSES,
    OutcomeClass,
    is_success,
)

_PACKAGED = SRC / "vllm_evolve" / "bench" / "schemas" / "eval_result.schema.json"
_ROOT = REPO_ROOT / "schemas" / "eval_result.schema.json"


def test_outcome_class_enum_matches_code():
    schema_vals = set(load_schema()["properties"]["outcome_class"]["enum"])
    code_vals = {c.value for c in OutcomeClass}
    assert schema_vals == code_vals, schema_vals ^ code_vals


def test_source_enum_is_real_smoke_and_frontier_only():
    # real_vllm = the only promotable source; local_smoke + frontier_sim are non-real (DoD-B only).
    assert set(load_schema()["properties"]["source"]["enum"]) == {
        "real_vllm", "local_smoke", "frontier_sim"}


def test_regime_enum_matches_shipped_profiles():
    import yaml
    profiles_dir = REPO_ROOT / "config" / "bench" / "profiles"
    shipped = set()
    for y in profiles_dir.glob("*.yaml"):
        doc = yaml.safe_load(y.read_text(encoding="utf-8"))
        shipped.add(doc["regime"])
    assert set(load_schema()["properties"]["regime"]["enum"]) == shipped, shipped


def test_two_schema_copies_do_not_drift():
    # load_schema() reads the PACKAGED copy; the repo-root copy is the doc-referenced one.
    # They must stay equal so an edit to one can't silently leave the other stale.
    assert json.loads(_PACKAGED.read_text(encoding="utf-8")) == json.loads(
        _ROOT.read_text(encoding="utf-8"))


def test_local_smoke_class_is_a_nonqualifying_failure():
    c = OutcomeClass.LOCAL_SMOKE_NONQUALIFYING
    assert c in FAILURE_CLASSES        # never a success/inconclusive
    assert is_success(c) is False      # structurally cannot be a clean pass


def test_simulator_class_is_a_nonqualifying_failure():
    c = OutcomeClass.SIMULATOR_NONQUALIFYING
    assert c in FAILURE_CLASSES        # frontier_sim result is never a success/inconclusive
    assert is_success(c) is False      # structurally cannot be a clean pass
