# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Shared retry policy for stateless tokenizer HTTP calls.

Callers own synchronous/asynchronous waiting and client lifetimes. HTTP status,
response validation, and inference requests are outside this policy.
"""

TOKENIZER_REQUEST_MAX_ATTEMPTS = 3
TOKENIZER_RETRY_BACKOFF_SECONDS = 0.1
