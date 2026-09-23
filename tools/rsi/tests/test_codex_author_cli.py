from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vllm_evolve.engine.codex_author_cli import author_with_codex


def _context() -> str:
    return json.dumps(
        {
            "doctrine": "real only",
            "spec": {},
            "diagnosis": {},
            "skeleton": "def schedule_batch(): ...",
            "parents": [],
            "peers": [],
            "lessons": [],
            "research_context": {},
            "last_errors": [],
            "budget": {},
            "generation": 0,
            "author_kind": "codex",
        }
    )


def test_codex_author_is_read_only_ephemeral_and_schema_bound():
    seen = {}
    response = {
        "source": "def schedule_batch():\n    pass\n",
        "manifest": {"mechanism_ids": ["gap"]},
    }

    def run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(response),
            stderr="",
        )

    assert author_with_codex(_context(), codex_bin="/codex", run=run) == response
    assert seen["argv"][:2] == ["/codex", "exec"]
    assert "--ephemeral" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--sandbox") + 1] == "read-only"
    assert "--output-schema" in seen["argv"]
    instruction = seen["argv"][-1]
    assert "diagnosis.required_change" in instruction
    assert "closed_loop_prefill_mass_restitution" in instruction
    assert "Prompt-Mass Restitution" in instruction
    assert seen["kwargs"]["input"] == _context()
    assert seen["kwargs"]["check"] is False


def test_codex_author_fails_closed_on_bad_output():
    def run(_argv, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="not-json", stderr="")

    with pytest.raises(ValueError, match="not one JSON object"):
        author_with_codex(_context(), codex_bin="/codex", run=run)


def test_codex_author_rejects_partial_context_before_invocation():
    with pytest.raises(ValueError, match="missing fields"):
        author_with_codex("{}", codex_bin="/codex")
