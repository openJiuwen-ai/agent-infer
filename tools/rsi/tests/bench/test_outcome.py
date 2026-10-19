"""Unit tests for the AC-9 single-label outcome taxonomy."""
from __future__ import annotations

from vllm_evolve.bench.outcome import OutcomeClass, is_success, resolve


def test_precedence_safety_beats_everything():
    present = {OutcomeClass.SAFETY_REJECTION, OutcomeClass.VLLM_CRASH, OutcomeClass.EVAL_RESULT}
    assert resolve(present) is OutcomeClass.SAFETY_REJECTION


def test_crash_beats_timeout_and_eval():
    present = {OutcomeClass.VLLM_CRASH, OutcomeClass.BENCH_TIMEOUT, OutcomeClass.EVAL_RESULT}
    assert resolve(present) is OutcomeClass.VLLM_CRASH


def test_high_variance_beats_eval_result():
    present = {OutcomeClass.HIGH_VARIANCE_INCONCLUSIVE, OutcomeClass.EVAL_RESULT}
    assert resolve(present) is OutcomeClass.HIGH_VARIANCE_INCONCLUSIVE


def test_eval_result_when_alone():
    assert resolve({OutcomeClass.EVAL_RESULT}) is OutcomeClass.EVAL_RESULT


def test_empty_set_is_unclassified():
    assert resolve(set()) is OutcomeClass.UNCLASSIFIED_FAILURE


def test_only_unclassified_present_stays_unclassified():
    assert resolve({OutcomeClass.UNCLASSIFIED_FAILURE}) is OutcomeClass.UNCLASSIFIED_FAILURE


def test_is_success_only_eval_result():
    assert is_success(OutcomeClass.EVAL_RESULT) is True
    assert is_success(OutcomeClass.HIGH_VARIANCE_INCONCLUSIVE) is False
    assert is_success(OutcomeClass.VLLM_CRASH) is False
