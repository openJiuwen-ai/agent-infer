"""Tests for the tools layer — the primary public API for agents."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def _enter_round_phase():
    """The legacy tool callables route through the phase guard, which (per AC-3)
    refuses to run from the pristine INIT phase. These tests exercise tool
    functions directly, so enter a real round phase first — mirroring how the
    tools are actually invoked inside a round. The guard's INIT-refusal is
    covered separately by test_phase_guard.py."""
    from vllm_evolve.tools import phase_guard

    phase_guard.reset_state()
    phase_guard.transition(phase_guard.Phase.READ_CONTEXT)
    yield
    phase_guard.reset_state()


# ─── Store tests ─────────────────────────────────────────────────

class TestStore:
    """Test SQLite store CRUD operations."""

    def _make_store(self):
        from vllm_evolve.store.db import Store
        return Store(tempfile.mktemp(suffix=".db"))

    def test_put_policy_idempotent(self):
        store = self._make_store()
        id1 = store.put_policy("def f(): pass", "f", "scheduling")
        id2 = store.put_policy("def f(): pass", "f", "scheduling")
        assert id1 == id2
        assert store.stats()["policies"] == 1

    def test_put_different_policies(self):
        store = self._make_store()
        id1 = store.put_policy("def f(): return 1", "f", "scheduling")
        id2 = store.put_policy("def f(): return 2", "f", "scheduling")
        assert id1 != id2
        assert store.stats()["policies"] == 2

    def test_put_eval(self):
        store = self._make_store()
        pid = store.put_policy("def f(): pass", "f", "scheduling")
        row_id = store.put_eval("policy", pid, "steady_low", "simulator",
                                fitness=0.55, metrics={"throughput": 1200})
        assert row_id > 0
        assert store.stats()["evaluations"] == 1

    def test_best_policies(self):
        store = self._make_store()
        pid1 = store.put_policy("def f(): return 1", "f", "scheduling")
        pid2 = store.put_policy("def f(): return 2", "f", "scheduling")
        store.put_eval("policy", pid1, "s", "simulator", fitness=0.5)
        store.put_eval("policy", pid2, "s", "simulator", fitness=0.8)
        best = store.best_policies("scheduling", n=1)
        assert len(best) == 1
        assert best[0]["policy_id"] == pid2

    def test_put_config(self):
        store = self._make_store()
        cid = store.put_config("Qwen", "a3", {"max_tokens": 8192})
        assert len(cid) == 16
        cfg = store.get_config(cid)
        assert cfg is not None
        assert json.loads(cfg["params_json"])["max_tokens"] == 8192

    def test_lineage_chain(self):
        store = self._make_store()
        seed_id = store.put_policy("def f(): return 0", "f", "scheduling")
        store.put_lineage(seed_id, generation=0, strategy="seed")
        child_id = store.put_policy("def f(): return 1", "f", "scheduling")
        store.put_lineage(child_id, parent_id=seed_id, generation=1, strategy="diff")
        chain = store.lineage_chain(child_id)
        assert len(chain) == 2
        assert chain[0]["policy_id"] == child_id
        assert chain[1]["policy_id"] == seed_id

    def test_put_check(self):
        store = self._make_store()
        pid = store.put_policy("def f(): pass", "f", "scheduling")
        store.put_check("policy", pid, True)
        store.put_check("policy", pid, False, ["L1: forbidden import os"])
        assert store.stats()["checks"] == 2


# ─── Verify tool tests ──────────────────────────────────────────

class TestVerifyTool:
    def test_safe_code_passes(self):
        from vllm_evolve.tools.verify_tool import verify_code
        r = verify_code("def schedule_batch(): pass", "scheduling", store_result=False)
        assert r["passed"] is True
        assert r["issues"] == []

    def test_unsafe_import_fails(self):
        from vllm_evolve.tools.verify_tool import verify_code
        r = verify_code("import os\ndef schedule_batch(): os.system('rm -rf /')",
                        "scheduling", store_result=False)
        assert r["passed"] is False
        assert any("L1" in i for i in r["issues"])

    def test_missing_function_fails(self):
        from vllm_evolve.tools.verify_tool import verify_code
        r = verify_code("def wrong_name(): pass", "scheduling", store_result=False)
        assert r["passed"] is False
        assert any("L2" in i or "Missing" in i for i in r["issues"])

    def test_bridge_runtime_type_redefinition_fails(self):
        from vllm_evolve.tools.verify_tool import verify_code

        code = (
            "class ScheduleDecision:\n"
            "    pass\n"
            "def schedule_batch():\n"
            "    pass\n"
        )
        r = verify_code(code, "scheduling", store_result=False)
        assert r["passed"] is False
        assert any("bridge-owned runtime class" in issue for issue in r["issues"])

    def test_config_valid(self):
        from vllm_evolve.tools.verify_tool import verify_config
        r = verify_config({"max_num_batched_tokens": 8192, "tensor_parallel_size": 4},
                          store_result=False)
        assert r["passed"] is True

    def test_config_invalid_range(self):
        from vllm_evolve.tools.verify_tool import verify_config
        r = verify_config({"gpu_memory_utilization": 2.0}, store_result=False)
        assert r["passed"] is False


# (Diff-tool tests removed: tools/diff_tool.py belonged to the deleted population-evolution
#  engine and was removed.)


# ─── Context tool tests ─────────────────────────────────────────

class TestContextTool:
    def test_build_context_empty_store(self):
        """Context should work even with empty store."""
        import tempfile

        from vllm_evolve.tools import store_tool
        # Use a fresh temp DB
        store_tool._store = None
        store_tool.get_store(tempfile.mktemp(suffix=".db"))

        from vllm_evolve.tools.context_tool import build_context
        ctx = build_context("scheduling")
        assert ctx["target_name"] == "scheduling"
        assert ctx["parent"] is None  # no policies yet
        assert ctx["inspirations"] == []

    def test_format_flat(self):
        from vllm_evolve.tools.context_tool import format_flat
        ctx = {
            "target_name": "scheduling",
            "parent": None,
            "inspirations": [],
            "seed_metrics": {},
            "recent_failures": [],
            "store_stats": {"policies": 0, "evaluations": 0},
        }
        text = format_flat(ctx)
        assert "scheduling" in text
        assert "CONTEXT" in text


# ─── Configure tool tests ───────────────────────────────────────

class TestConfigureTool:
    def test_generate_candidates(self):
        from vllm_evolve.tools.configure_tool import generate_candidates
        candidates = generate_candidates(n=5)
        assert len(candidates) > 0
        assert "max_num_batched_tokens" in candidates[0]

    def test_candidates_respect_constraints(self):
        from vllm_evolve.tools.configure_tool import generate_candidates
        candidates = generate_candidates(n=50, seed=42)
        for c in candidates:
            if c.get("tensor_parallel_size") == 8:
                # TP=8 → PP should be 1 (constraint)
                assert c.get("pipeline_parallel_size", 1) == 1 or "pipeline_parallel_size" not in c

    def test_export_vllm_yaml(self):
        from vllm_evolve.tools.configure_tool import export_vllm_yaml
        yaml_str = export_vllm_yaml(
            {"max_num_batched_tokens": 8192, "enable_chunked_prefill": True}
        )
        assert "max_num_batched_tokens" in yaml_str
        assert "8192" in yaml_str


# TestSimulateTool, TestEvolutionIntegration, TestAdapterPlugin removed (M3):
# they exercised the deleted DES simulator / old V0 engine.adapter template.
# The V1 plugin is covered by tests/bench/test_plugin_template.py; the live
# evolve loop is now GPU-gated. See tests/REMOVED_SIM_TESTS.md.


# ─── Error handling tests ────────────────────────────────────────

class TestErrorHandling:
    """Test that tools handle errors gracefully."""

    def test_verify_syntax_error(self):
        from vllm_evolve.tools.verify_tool import verify_code
        result = verify_code("def schedule_batch(: broken syntax", "scheduling", store_result=False)
        assert result["passed"] is False

    def test_store_concurrent_safe(self):
        """Two stores pointing to same DB should not corrupt."""
        import tempfile

        from vllm_evolve.store.db import Store
        db_path = tempfile.mktemp(suffix=".db")
        s1 = Store(db_path)
        s2 = Store(db_path)
        s1.put_policy("def f(): return 1", "f", "scheduling")
        s2.put_policy("def f(): return 2", "f", "scheduling")
        # Both should succeed without corruption
        assert s1.stats()["policies"] >= 1
        assert s2.stats()["policies"] >= 1
        s1.close()
        s2.close()

    def test_configure_zero_budget(self):
        from vllm_evolve.tools.configure_tool import generate_candidates
        candidates = generate_candidates(n=0)
        assert candidates == []


# Trace loader tests removed (M3): sim.trace_loader deleted. Real traces are
# loaded by the bench dataset adapters (tests/bench/test_datasets.py).


# (TestTrustChainOverfit removed: trust/chain.py TrustChain belonged to the deleted
#  population-evolution engine. The live anti-overfit floor check is in the bench/autopt
#  verify-gain holdout path.)


# GPUTimingModel + multi-scenario tests removed (M3): sim.pd /
# sim.scenario_registry deleted. Bench load/dataset tests cover scenarios
# (tests/bench/test_load.py, test_datasets.py). See tests/REMOVED_SIM_TESTS.md.
