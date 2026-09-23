"""M2: read-only `ve inspect` verb + the ve-research CITED-FACT contract. A research sub-agent is an
untrusted producer, so the orchestrator must be able to verify its citations against the store — a
fabricated policy_id is DETECTED, never trusted. Pure, no GPU, no store mutation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.cli import main as cli_main  # noqa: E402
from vllm_evolve.ops.inspect import inspect_top, verify_citations  # noqa: E402
from vllm_evolve.store.db import Store  # noqa: E402

_POLICY = "def schedule_batch(running, state):\n    return None\n"


def _store_with_one_policy(tmp_path) -> tuple[Store, str]:
    store = Store(str(tmp_path / "t.db"))
    pid = store.put_policy(_POLICY, "schedule_batch", "scheduling")
    return store, pid


def test_verify_citations_detects_fabricated(tmp_path):
    store, pid = _store_with_one_policy(tmp_path)
    res = verify_citations(store, [pid, "notarealpolicy00"])
    assert res["real"] == [pid] and res["fabricated"] == ["notarealpolicy00"]
    assert res["all_real"] is False and res["checked"] == 2
    assert verify_citations(store, [pid])["all_real"] is True
    vacuous = verify_citations(store, [])                          # nothing cited
    assert vacuous["all_real"] is True and vacuous["checked"] == 0  # caller must check checked>0


def test_inspect_top_is_read_only(tmp_path):
    store, _ = _store_with_one_policy(tmp_path)
    before = store.stats()
    view = inspect_top(store, "scheduling", 5)
    assert view["view"] == "top"
    assert store.stats() == before                               # SELECT only, no mutation


def test_ar_inspect_cli_verify_citations(tmp_path, capsys):
    _, pid = _store_with_one_policy(tmp_path)
    rc = cli_main.main(["inspect", "--verify-citations", f"{pid},notarealpolicy00",
                      "--db", str(tmp_path / "t.db")])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] is True
    assert pid in out["real"] and out["fabricated"] == ["notarealpolicy00"]


def test_ar_inspect_needs_a_selector(tmp_path, capsys):
    _store_with_one_policy(tmp_path)                              # so --db exists
    rc = cli_main.main(["inspect", "--db", str(tmp_path / "t.db")])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "inspect_needs_a_selector"


def test_ar_inspect_refuses_nonexistent_db_and_creates_nothing(tmp_path, capsys):
    # a read verb must NEVER materialize a stray empty store (Codex/skeptic M2 finding).
    missing = tmp_path / "nope.db"
    rc = cli_main.main(["inspect", "--verify-citations", "x", "--db", str(missing)])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 2 and out["outcome_class"] == "store_not_found"
    assert not missing.exists()
