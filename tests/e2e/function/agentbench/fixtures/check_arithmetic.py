# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Run with Python directly; no third-party test dependencies are needed."""

import unittest

from arithmetic import add


class AdditionChecks(unittest.TestCase):
    def test_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)

    def test_negative_numbers(self):
        self.assertEqual(add(-2, -3), -5)

    def test_zero(self):
        self.assertEqual(add(5, 0), 5)


if __name__ == "__main__":
    unittest.main()
