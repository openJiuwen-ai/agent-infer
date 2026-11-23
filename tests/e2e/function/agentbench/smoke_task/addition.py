# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""A deliberately broken addition function for the AgentBench smoke task."""


def add(a, b):
    """Return the sum of two numbers."""
    return a - b


if __name__ == "__main__":
    assert add(2, 3) == 5
    assert add(-2, -3) == -5
    assert add(5, 0) == 5
