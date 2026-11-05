# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Shared, process-isolated model service for CPU contract tests."""

from .process import MockServerProcess

__all__ = ["MockServerProcess"]
