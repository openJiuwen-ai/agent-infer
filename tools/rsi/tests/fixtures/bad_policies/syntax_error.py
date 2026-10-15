# Fixture category: syntax_error (L1 rejection).
# ``ast.parse`` raises SyntaxError; check_safety catches it before signatures.
from __future__ import annotations


def schedule_batch(waiting_requests, running_requests
    # missing closing paren on purpose
    return None
