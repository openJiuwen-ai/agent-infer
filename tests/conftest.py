import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "cpu_test: tests that run without GPU")
