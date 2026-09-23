"""M5 / AC5: the full sub-agent-orchestrated autopt loop runs off-box and can ONLY conclude DoD-B.

run_autopt is driven with the ve-author bridge (author_evolve_fn). The bridge benches via
local_smoke (marker_verified always False), so the authored policy can never adopt — the
loop reaches the code/author path and still concludes DoD-B (no provable gain). Off-GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(SRC))

from vllm_evolve.core.schemas import SCHEDULING_QUEUE, Profile, Spec  # noqa: E402
from vllm_evolve.engine.diagnose import diagnose  # noqa: E402
from vllm_evolve.engine.evolve_target import author_evolve_fn  # noqa: E402
from vllm_evolve.engine.orchestrate import run_autopt  # noqa: E402


def _scheduling_queue_profile(_cfg) -> Profile:
    # a deterministic stub measurement that diagnoses SCHEDULING_QUEUE (free SM capacity +
    # a standing queue, KV not full, no preemption) so the loop reaches the code/author target.
    # Stands in for the real bench in an off-box unit test; constant metrics -> no config gain.
    return Profile(metrics={"output_throughput_tok_s": 10.0},
                   gpu={"sm_util_max": 70.0, "duty_cycle": 0.5, "mem_bw_util": 30.0,
                        "mem_used_mb": 1000.0, "mem_total_mb": 40000.0},
                   vllm={"kv_util": 0.4, "waiting": 50, "running": 4, "preempt": 0},
                   per_seed={"output_throughput_tok_s": [10.0, 10.0, 10.0]},
                   marker_verified=True, source="real_vllm")


def test_stub_profile_diagnoses_scheduling_queue():
    # sanity: the stub really reaches the only regime where the code/author target is ranked in.
    assert diagnose(_scheduling_queue_profile(None)).bottleneck == SCHEDULING_QUEUE


_AUTHOR_CALLS: list[bool] = []


def _stub_author(_base_config, _spec):
    _AUTHOR_CALLS.append(True)        # sentinel: prove the bridge actually invoked the author
    return "def schedule_batch(running, state):\n    return None\n"


def test_subagent_orchestrated_loop_concludes_dod_b():
    _AUTHOR_CALLS.clear()
    evolve_fn = author_evolve_fn(_stub_author)        # default local_smoke -> always unverified
    spec = Spec(metric="output_throughput_tok_s", direction="max")
    res = run_autopt(spec, eval_fn=_scheduling_queue_profile, evolve_fn=evolve_fn,
                     base_config={"n_requests": 50, "model": "facebook/opt-125m"}, max_rounds=3)
    assert res["outcome"] == "dod_b" and res["adopted"] is None
    rounds = res["rounds"]
    # the CODE/author path actually RAN the bridge (author invoked) and its output was rejected as
    # UNVERIFIED (local_smoke can't verify) — not merely "picked" (skeptic M5: searched_target alone
    # is set before the verified-candidate check, so it would pass even if the bridge were absent).
    code_round = next(r for r in rounds if r.get("searched_target") == "code:schedule_batch")
    assert _AUTHOR_CALLS, "the bridge must actually invoke the ve-author"
    assert code_round["verdict"]["verdict"] == "no_verified_candidate"
    # the CONFIG path was ALSO blocked (no lock silently regressed into an adoption).
    cfg_round = next(r for r in rounds if r.get("searched_target") == "config:max_num_seqs")
    assert cfg_round.get("same_caliber") != "ok"
