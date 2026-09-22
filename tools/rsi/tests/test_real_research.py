from __future__ import annotations

from vllm_evolve.engine.real_research import _gap_cards, _select_gap_cards
from vllm_evolve.knowledge.schemas import MechanismCard


def test_real_scheduler_cards_are_live_gap_complete_and_structurally_distinct():
    cards = [
        MechanismCard.from_dict(value)
        for value in _gap_cards("github:vllm-main:abc")
    ]
    assert len(cards) == 12
    assert len({card.mechanism_id for card in cards}) == 12
    assert all(card.validate_live_gap() == [] for card in cards)
    assert {
        tuple(card.expected_action_counters)
        for card in cards
    } == {
        (
            "policy_invocation_count",
            "deferred_request_actions",
            "reorder_action_count",
        ),
        ("policy_invocation_count", "preemption_count"),
        (
            "policy_invocation_count",
            "preemption_count",
            "reorder_action_count",
        ),
        ("policy_invocation_count", "reorder_action_count"),
    }
    assert all(
        "github:vllm-main:abc" in card.source_ids
        for card in cards
    )
    bounded = {
        "workset_preserving_slo_exchange",
        "decode_mass_anchored_rescue_wave",
        "budget_feedback_fill_parity_controller",
    }
    assert all(
        "O(k)" in card.complexity_cost
        and "fixed bounded FCFS head window" in card.complexity_cost
        for card in cards
        if card.mechanism_id in bounded
    )
    assert {
        "recoverable_short_residency_frontier",
        "slack_guarded_first_token_wavefront",
        "slo_yield_per_residency_frontier",
        "adaptive_slo_cliff_switch",
        "workset_preserving_slo_exchange",
        "decode_mass_anchored_rescue_wave",
        "budget_feedback_fill_parity_controller",
        "lifecycle_gated_one_shot_rescue",
    }.issubset({card.mechanism_id for card in cards})


def test_real_scheduler_cards_keep_falsified_mechanisms_as_evidence_only():
    cards = _select_gap_cards(
        "github:vllm-main:abc",
        {
            "falsified_hypotheses": [
                {"mechanism_id": "bounded_first_fit_head_bypass"},
                {"mechanism_id": "max_cardinality_kv_fit_frontier"},
                {"mechanism_id": "one_shot_residual_exchange"},
                {"mechanism_id": "completion_protected_residual_exchange"},
                {"mechanism_id": "recoverable_short_residency_frontier"},
                {"mechanism_id": "slack_guarded_first_token_wavefront"},
                {"mechanism_id": "slo_yield_per_residency_frontier"},
                {"mechanism_id": "adaptive_slo_cliff_switch"},
            ]
        },
    )
    assert [card["mechanism_id"] for card in cards] == [
        "workset_preserving_slo_exchange",
        "decode_mass_anchored_rescue_wave",
        "budget_feedback_fill_parity_controller",
        "lifecycle_gated_one_shot_rescue",
    ]
