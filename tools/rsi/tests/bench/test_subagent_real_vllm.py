"""AC7/AC8 (box-gated, NON-gating): the autopt SUB-AGENT layer on REAL vLLM.

These exercise the ve-author bridge + the sub-agent-orchestrated loop against ACTUAL vLLM. They are
GPU-gated: off-hardware they are SKIPPED (never a fabricated pass); on a GPU host
(``VLLM_EVOLVE_GPU=1``) they run the real serve+bench path. The off-box suite proves the NEGATIVE
(an authored / local_smoke policy can never adopt); these prove the POSITIVE only where it can be
measured for real — and any adoption goes through the same frozen gates, never a shortcut.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

_BASE = {"model": "facebook/opt-125m", "n_requests": 8, "max_seeds": 1}


@pytest.mark.requires_gpu
def test_ac7_author_bridge_verifies_on_real_vllm():  # pragma: no cover - GPU host only
    # On a real box the ve-author bridge benches the authored policy on ACTUAL vLLM; marker_verified
    # comes from that real run (NOT local_smoke). The seed policy genuinely runs the plugin, so the
    # real bench verifies it and the candidate carries the policy PATH for the re-bench.
    from vllm_evolve.core.schemas import Spec
    from vllm_evolve.engine.evolve_target import author_evolve_fn
    from vllm_evolve.engine.profile import collect_profile
    seed = (REPO_ROOT / "targets" / "scheduling" / "seed.py").read_text(encoding="utf-8")
    evolve_fn = author_evolve_fn(lambda _c, _s: seed, profile_fn=collect_profile)
    cand = evolve_fn(dict(_BASE), Spec(metric="output_throughput_tok_s", direction="max"))
    assert cand.marker_verified is True              # the REAL bench verified the plugin ran
    assert cand.value.get("policy")                  # carries the policy PATH (real-gate re-bench)


@pytest.mark.requires_gpu
def test_ac8_orchestrated_loop_on_real_vllm_never_fabricates():  # pragma: no cover - GPU host only
    # The sub-agent-orchestrated loop on REAL vLLM, fed real authored policies via the bridge. It
    # exercises the full real path and concludes DoD-B HONESTLY: adoption additionally requires a
    # MEASURED served-model quality non-regression (quality_measure_fn), which is box-gated future
    # work — with none wired, NO candidate can adopt and the loop must NOT fabricate one (this
    # mirrors ar_cli.py cmd_autopt, which also passes no quality_measure_fn yet).
    from vllm_evolve.core.schemas import Spec
    from vllm_evolve.engine.evolve_target import author_evolve_fn
    from vllm_evolve.engine.orchestrate import run_autopt
    from vllm_evolve.engine.profile import collect_profile
    seed = (REPO_ROOT / "targets" / "scheduling" / "seed.py").read_text(encoding="utf-8")
    evolve_fn = author_evolve_fn(lambda _c, _s: seed, profile_fn=collect_profile)
    spec = Spec(metric="output_throughput_tok_s", direction="max")
    res = run_autopt(spec, eval_fn=collect_profile, evolve_fn=evolve_fn,
                     base_config={**_BASE, "max_num_seqs": 8}, max_rounds=2)
    assert res["outcome"] == "dod_b" and res["adopted"] is None   # never a fabricated adoption
    searched = [r.get("searched_target") for r in res["rounds"] if r.get("searched_target")]
    assert searched                                  # the loop actually searched on real vLLM
