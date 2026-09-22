# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Taxonomy completeness and explicit unimplemented collection status."""

import json
import unittest

from agentinfer.rsi.feedback import TAXONOMY, Backend, Layer, list_layers


class TaxonomyTests(unittest.TestCase):
    def test_layers_match_the_approved_logical_boundaries(self):
        self.assertEqual(
            {layer.value for layer in Layer},
            {
                "api_server",
                "engine",
                "worker",
                "model_scripts",
                "parallel",
                "ops.communication",
                "ops.compute",
            },
        )
        self.assertEqual(set(TAXONOMY), set(Layer))

    def test_every_backend_has_actionable_plans_without_claiming_connected_probes(self):
        for backend in Backend:
            rows = list_layers(backend)
            self.assertEqual(len(rows), 7)
            json.dumps(rows)
            for row in rows:
                self.assertEqual(row["backend"], backend.value)
                self.assertEqual(row["collection_status"], "planned_not_connected")
                self.assertTrue(row["description"])
                self.assertTrue(row["signals"])
                self.assertTrue(row["validators"])
                self.assertTrue(row["collection_hint"])
        self.assertNotEqual(list_layers("cuda")[-1]["collection_hint"], list_layers("ascend")[-1]["collection_hint"])

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            list_layers("cpu")


if __name__ == "__main__":
    unittest.main()
