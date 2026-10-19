"""P4-2: intent router — pure NL -> routing PROPOSAL (no measure/judge/run). No GPU."""
from __future__ import annotations

from vllm_evolve.intent.router import MODES, detect_mode, route


def test_modes_are_exactly_the_three_top_level():
    assert MODES == ("autopt", "tune", "port")          # research is NOT a mode


def test_goal_text_routes_to_autopt():
    p = route("maximize goodput under 500ms p99")
    assert p["mode"] == "autopt"
    assert p["target"] == "scheduling" and p["target_supported"] is True
    assert p["spec"]["metric"] and p["policy"] is None


def test_policy_path_routes_to_tune():
    p = route("tune this policy targets/scheduling/work.py for throughput")
    assert p["mode"] == "tune"
    assert p["policy"] == "targets/scheduling/work.py"


def test_version_hardware_routes_to_port():
    assert detect_mode("adapt work.py across vLLM versions", has_policy=True) == "port"
    p = route("port the scheduler to vLLM 0.22 on h100 hardware")
    assert p["mode"] == "port"          # port keywords win even with a policy present


def test_unsupported_target_is_flagged_not_raised():
    p = route("maximize goodput", target="kv_eviction")
    assert p["target_supported"] is False
    assert p["target_rejection"]["outcome_class"] == "unsupported_target"


def test_route_is_pure_proposal_no_verdict_fields():
    p = route("maximize goodput")
    # a PROPOSAL never carries a measurement/verdict
    for forbidden in ("verdict", "eval_result", "gain", "accepted", "kept"):
        assert forbidden not in p
