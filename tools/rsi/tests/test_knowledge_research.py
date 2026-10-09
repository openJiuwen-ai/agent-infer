"""P4-6: store-backed knowledge synthesis for ve-research (cited facts; no RAG). No GPU."""
from __future__ import annotations

from vllm_evolve.knowledge import gather
from vllm_evolve.store.db import Store


def _seed(db_path) -> str:
    store = Store(db_path)
    try:
        pid = store.put_policy("def schedule_batch(r, s):\n    return None\n",
                               "schedule_batch", "scheduling")
        store.put_lineage(pid, parent_id=None, generation=0, strategy="seed")
        store.put_eval("policy", pid, "throughput", "real_vllm", fitness=0.62,
                       metrics={"goodput_req_s": 12.0})
        return pid
    finally:
        store.close()


def test_gather_returns_cited_prior_attempts(tmp_path):
    db = str(tmp_path / "k.db")
    pid = _seed(db)
    facts = gather("scheduling", db=db, n=5)
    assert facts["target"] == "scheduling"
    assert any(p.get("policy_id") == pid for p in facts["top_policies"])
    # citations are real ids the agent must quote (policy + eval), never free-form
    assert f"policy:{pid}" in facts["citations"]
    assert any(c.startswith("eval:") for c in facts["citations"])
    assert pid in facts["lineage"] and pid in facts["histories"]


def test_gather_empty_store_is_well_formed(tmp_path):
    facts = gather("scheduling", db=str(tmp_path / "empty.db"), n=5)
    assert facts["top_policies"] == [] and facts["citations"] == []
    assert facts["target"] == "scheduling"
