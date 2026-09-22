# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agentinfer.rsi.knowledge import KnowledgeRepository
from agentinfer.rsi.storage import ConflictError


def proposal(**overrides):
    return dict(
        namespace="vllm.ops.compute",
        scope="project-a",
        backend="cuda",
        component_version="engine-1",
        workload="long-prefill",
        title="Kernel case",
        claim="A hypothesis, not a proven optimization.",
        evidence=[],
        **overrides,
    )


def query(**overrides):
    fields = {"scope": "project-a", "backend": "cuda", "component_version": "engine-1", "workload": "long-prefill"}
    return {**fields, **overrides}


class TestKnowledge(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.knowledge = KnowledgeRepository(Path(temporary.name) / "knowledge.sqlite")

    def test_exact_scope_and_literal_search(self):
        knowledge = self.knowledge
        record = knowledge.propose(proposal())
        self.assertEqual(knowledge.search(**query(), text="hypothesis")[0]["knowledge_id"], record["knowledge_id"])
        for field, value in (
            ("scope", "project-b"),
            ("backend", "ascend"),
            ("component_version", "engine-2"),
            ("workload", "short"),
        ):
            self.assertEqual(knowledge.search(**query(**{field: value})), [])
        self.assertEqual(knowledge.search(**query(), text="%"), [])
        with self.assertRaisesRegex(ValueError, "explicit"):
            knowledge.search(**query(scope="*"))
        with self.assertRaisesRegex(ValueError, "explicit"):
            knowledge.propose({**proposal(), "component_version": ""})

    def test_proposal_cannot_self_validate(self):
        knowledge = self.knowledge
        with self.assertRaisesRegex(ValueError, "self-validate"):
            knowledge.propose(proposal(status="validated"))
        record = knowledge.propose(proposal())
        with self.assertRaisesRegex(ValueError, "trusted experimental"):
            knowledge.transition(
                record["knowledge_id"], expected_revision=0, status="validated", reason="LLM says PASS"
            )
        self.assertEqual(knowledge.get(record["knowledge_id"])["status"], "draft")

    def test_review_needs_provenance_and_history_is_preserved(self):
        knowledge = self.knowledge
        record = knowledge.propose(proposal())
        args = {
            "knowledge_id": record["knowledge_id"],
            "expected_revision": 0,
            "status": "reviewed",
            "reason": "source inspected",
        }
        with self.assertRaisesRegex(ValueError, "source_reviewed"):
            knowledge.transition(**args)
        reviewed = knowledge.transition(
            **args, evidence=[{"ref": "git:fixed-revision:file", "outcome": "source_reviewed"}]
        )
        self.assertEqual(reviewed["status"], "reviewed")
        with self.assertRaisesRegex(ConflictError, "stale"):
            knowledge.transition(record["knowledge_id"], expected_revision=0, status="stale", reason="version changed")
        knowledge.transition(record["knowledge_id"], expected_revision=1, status="stale", reason="version changed")
        self.assertEqual(knowledge.search(**query()), [])
        self.assertEqual(len(knowledge.search(**query(), status="stale")), 1)
        self.assertEqual(
            [item["status"] for item in knowledge.history(record["knowledge_id"])], ["draft", "reviewed", "stale"]
        )

    def test_seed_namespaces_are_demo_drafts_and_idempotent(self):
        knowledge = self.knowledge
        seeds = knowledge.seed_demo()
        self.assertEqual(len(seeds), 22)
        self.assertEqual(knowledge.seed_demo(), seeds)
        names = {item["namespace"] for item in seeds}
        self.assertTrue({"system", "semantic-router", "router", "scheduling", "harness", "agent-cache"} <= names)
        for engine in ("vllm", "vllm-ascend"):
            for layer in (
                "api_server",
                "engine",
                "worker",
                "model_scripts",
                "parallel",
                "ops.communication",
                "ops.compute",
            ):
                self.assertIn(f"{engine}.{layer}", names)
        self.assertTrue(all(item["demo"] and item["status"] == "draft" and not item["evidence"] for item in seeds))
        cuda = knowledge.search(scope="demo", backend="cuda", component_version="demo-unverified", workload="demo")
        self.assertTrue(cuda and all(not item["namespace"].startswith("vllm-ascend") for item in cuda))
