# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import pytest  # noqa: F401


def pytest_configure(config):
    config.addinivalue_line("markers", "cpu_test: tests that run without GPU")
    config.addinivalue_line("markers", "gpu_test: tests that require GPU")
    config.addinivalue_line("markers", "e2e_perf: end-to-end performance benchmarks")
