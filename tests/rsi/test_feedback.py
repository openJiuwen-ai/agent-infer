# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""CPU-only acceptance checks for evidence integrity and experiment separation."""

import json
import tempfile
import unittest
from pathlib import Path

from agentinfer.rsi.feedback import FeedbackRecord, UnavailableProbe, evaluate_feedback, load_records


def scope():
    return {
        "candidate_id": "candidate-1",
        "baseline_id": "baseline-1",
        "backend": "cuda",
        "engine_version": "vllm-pinned-commit",
        "model_revision": "model-pinned-revision",
        "workload_id": "fixed-workload-sha256",
    }


def record(**overrides):
    data = {
        "check_id": "kernel-reference",
        "hypothesis_id": "reduce-normalization-redundancy",
        "candidate_id": "candidate-1",
        "baseline_id": "baseline-1",
        "backend": "cuda",
        "layer": "ops.compute",
        "category": "correctness",
        "verdict": "pass",
        "metric": "mismatched_elements",
        "value": 0,
        "unit": "elements",
        "scope": {
            key: value for key, value in scope().items() if key not in ("candidate_id", "baseline_id", "backend")
        },
        "diagnostic": False,
        "evidence_refs": ["artifacts/kernel-reference.json"],
        "next_test": "Run the fixed integration workload on the target backend.",
    }
    data.update(overrides)
    return FeedbackRecord.from_dict(data)


class FeedbackTests(unittest.TestCase):
    def test_record_json_roundtrip_and_scope_snapshot(self):
        source = record().to_dict()
        normalized = FeedbackRecord.from_dict(source)
        source["scope"]["engine_version"] = "mutated"
        self.assertEqual(normalized.scope["engine_version"], "vllm-pinned-commit")
        self.assertEqual(FeedbackRecord.from_dict(json.loads(json.dumps(normalized.to_dict()))), normalized)

    def test_valid_external_validator_record_passes(self):
        report = evaluate_feedback([record()], ["kernel-reference"], scope())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["checks"][0]["suggested_validators"][0], "high_precision_reference")

    def test_frozen_required_checks_cannot_be_satisfied_by_unrelated_passes(self):
        report = evaluate_feedback([record()], ["missing-holdout"], scope())
        self.assertEqual(report["status"], "INCONCLUSIVE")
        self.assertEqual(report["ignored_check_ids"], ["kernel-reference"])
        self.assertIn("unavailable", report["checks"][0]["reason"])

    def test_known_failure_dominates_missing_evidence(self):
        report = evaluate_feedback([record(verdict="fail", value=7)], ["kernel-reference", "holdout"], scope())
        self.assertEqual(report["status"], "FAIL")

    def test_profiled_performance_is_not_acceptance(self):
        for verdict in ("pass", "fail"):
            with self.subTest(verdict=verdict):
                measured = record(category="performance", diagnostic=True, verdict=verdict)
                report = evaluate_feedback([measured], ["kernel-reference"], scope())
                self.assertEqual(report["status"], "INCONCLUSIVE")
                self.assertIn("without profiling", report["checks"][0]["next_test"])
        report = evaluate_feedback([record(diagnostic=True)], ["kernel-reference"], scope())
        self.assertEqual(report["status"], "PASS")

    def test_backend_identity_and_environment_mismatches_are_not_mixed(self):
        for name in ("candidate_id", "baseline_id", "backend", "engine_version", "model_revision", "workload_id"):
            with self.subTest(name=name):
                expected = scope()
                expected[name] = "ascend" if name == "backend" else "another-value"
                report = evaluate_feedback([record()], ["kernel-reference"], expected)
                self.assertEqual(report["status"], "INCONCLUSIVE")
        expanded = scope() | {"dtype": "bf16"}
        self.assertEqual(evaluate_feedback([record()], ["kernel-reference"], expanded)["status"], "INCONCLUSIVE")

    def test_duplicate_check_results_require_explicit_trial_aggregation(self):
        report = evaluate_feedback([record(), record()], ["kernel-reference"], scope())
        self.assertEqual(report["status"], "INCONCLUSIVE")
        self.assertIn("duplicate", report["checks"][0]["reason"])

    def test_missing_evidence_and_inconclusive_verdict_do_not_pass(self):
        for changes in ({"evidence_refs": []}, {"verdict": "inconclusive", "value": None}, {"synthetic": True}):
            with self.subTest(changes=changes):
                report = evaluate_feedback([record(**changes)], ["kernel-reference"], scope())
                self.assertEqual(report["status"], "INCONCLUSIVE")

    def test_unimplemented_probe_returns_unavailable_not_zero(self):
        probe = UnavailableProbe(
            "transfer-overlap", "overlap-hypothesis", "ops.communication", "performance", "overlap", "ratio"
        )
        measured = probe.collect(scope())
        self.assertIsNone(measured[0].value)
        self.assertEqual(measured[0].verdict.value, "unavailable")
        self.assertEqual(evaluate_feedback(measured, ["transfer-overlap"], scope())["status"], "INCONCLUSIVE")
        with self.assertRaises(ValueError):
            record(verdict="unavailable", value=0)

    def test_invalid_record_fields_are_rejected(self):
        cases = [
            {"hypothesis_id": " "},
            {"baseline_id": ""},
            {"candidate_id": ""},
            {"backend": "cpu"},
            {"layer": "kernel"},
            {"category": "quality"},
            {"verdict": "probably"},
            {"metric": ""},
            {"unit": ""},
            {"next_test": ""},
            {"value": float("nan")},
            {"value": float("inf")},
            {"value": -float("inf")},
            {"value": True},
            {"value": "1"},
            {"value": None},
            {"diagnostic": "false"},
            {"synthetic": 1},
            {"evidence_refs": "artifact.json"},
            {"evidence_refs": [""]},
            {"scope": {}},
            {"scope": {"engine_version": "", "model_revision": "x", "workload_id": "x"}},
            {"unexpected": "field"},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                record(**changes)

    def test_invalid_frozen_evaluation_configuration_is_rejected(self):
        for checks in ([], ["x", "x"], [""], "kernel-reference", {"kernel-reference": {}}, 1, None):
            with self.subTest(checks=checks), self.assertRaises(ValueError):
                evaluate_feedback([record()], checks, scope())
        with self.assertRaises(ValueError):
            evaluate_feedback([record()], ["kernel-reference"], scope() | {"backend": "cpu"})

    def test_loader_preserves_envelope_synthetic_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.json"
            payload = {
                "synthetic": True,
                "description": "Demo only",
                "scope": scope(),
                "required_checks": ["kernel-reference"],
                "records": [record().to_dict()],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            measured = load_records(path)
            self.assertTrue(measured[0].synthetic)
            self.assertEqual(
                evaluate_feedback(measured, payload["required_checks"], payload["scope"])["status"], "INCONCLUSIVE"
            )
            path.write_text(json.dumps([record().to_dict()]), encoding="utf-8")
            self.assertEqual(len(load_records(path)), 1)
            path.write_text('{"records": "invalid"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_records(path)

    def test_bundled_demo_is_explicitly_inconclusive(self):
        path = Path(__file__).resolve().parents[2] / "examples" / "rsi" / "feedback.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(payload["synthetic"])
        result = evaluate_feedback(load_records(path), payload["required_checks"], payload["scope"])
        self.assertEqual(result["status"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
