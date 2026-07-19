"""Phase 1 boundary tests for interaction handlers and controller."""

from __future__ import annotations

import pytest

from agentcache.benchmarks.agents.claude.interaction import (
    InteractionContext,
    InteractionController,
    InteractionState,
    NoProgressHandler,
    PlanApprovalHandler,
    StartupDialogHandler,
    YesConfirmationHandler,
)


@pytest.mark.parametrize(
    "terminal",
    [
        "Do you trust the files in this directory?",
        "Security Guide — press Enter to continue",
        "Choose a theme:",
    ],
)
def test_startup_handler_recognizes_dialogs(terminal: str) -> None:
    handler = StartupDialogHandler()
    state = InteractionState()
    ctx = InteractionContext(terminal, None, 0.0, state)
    assert handler.recognize(ctx) is True


def test_startup_handler_sends_enter_for_dialog() -> None:
    handler = StartupDialogHandler()
    state = InteractionState()
    ctx = InteractionContext("Do you trust?", None, 0.0, state)
    action = handler.act(ctx)
    assert action.kind == "send_keys"
    assert action.keys == "Enter"


def test_startup_handler_dismisses_when_prompt_visible() -> None:
    handler = StartupDialogHandler()
    state = InteractionState()
    ctx = InteractionContext("some output\n❯ ", None, 0.0, state)
    assert handler.recognize(ctx) is True
    action = handler.act(ctx)
    assert action.kind == "none"
    assert state.startup_dismissed is True


@pytest.mark.parametrize("terminal", ["❯ Yes", "❯ 1. Yes", "> Yes"])
def test_yes_handler_recognizes_yes_prompts(terminal: str) -> None:
    handler = YesConfirmationHandler()
    state = InteractionState()
    ctx = InteractionContext(terminal, None, 0.0, state)
    assert handler.recognize(ctx) is True


def test_controller_suppresses_same_yes_prompt_across_dynamic_updates() -> None:
    controller = InteractionController((YesConfirmationHandler,))
    state = InteractionState(startup_dismissed=True)

    assert controller.process(InteractionContext("❯ 1. Yes\n⠋ Working", None, 1.0, state)).kind == "send_keys"
    assert controller.process(InteractionContext("❯ 1. Yes\n⠙ Working", None, 1.5, state)).kind == "none"
    assert controller.process(InteractionContext("Working", None, 2.0, state)).kind == "none"
    assert controller.process(InteractionContext("❯ 1. Yes", None, 2.5, state)).kind == "send_keys"


def test_no_progress_handler_requests_one_check_per_idle_period() -> None:
    handler = NoProgressHandler()
    state = InteractionState()
    transcript: list[dict[str, object]] = [{"type": "assistant"} for _ in range(20)]

    assert handler.recognize(InteractionContext("❯", transcript, 100.0, state)) is False
    assert handler.recognize(InteractionContext("❯", transcript, 109.9, state)) is False
    assert handler.recognize(InteractionContext("❯", transcript, 110.0, state)) is True
    assert handler.act(InteractionContext("❯", transcript, 110.0, state)).kind == "check_workspace"
    assert handler.recognize(InteractionContext("❯", transcript, 120.0, state)) is False
    assert handler.recognize(InteractionContext("Working", transcript, 121.0, state)) is False
    assert handler.recognize(InteractionContext("❯", transcript, 122.0, state)) is False
    assert handler.recognize(InteractionContext("❯", transcript, 132.0, state)) is True


def test_no_progress_handler_disabled_in_plan_subagent() -> None:
    handler = NoProgressHandler(enable_idle_detection=False)
    transcript: list[dict[str, object]] = [{"type": "assistant"} for _ in range(30)]

    assert handler.recognize(InteractionContext("❯", transcript, 120.0, InteractionState())) is False


def test_plan_prompt_suppression_ignores_dynamic_updates_and_resets_after_disappearance() -> None:
    controller = InteractionController(
        (PlanApprovalHandler, YesConfirmationHandler),
    )
    state = InteractionState(startup_dismissed=True)
    plan = "Claude has written up a plan and is ready to execute.\n❯ Yes, and bypass permissions"

    assert controller.process(InteractionContext(plan + "\n⠋", None, 1.0, state)).kind == "send_keys"
    assert controller.process(InteractionContext(plan + "\n⠙", None, 1.5, state)).kind == "none"
    assert controller.process(InteractionContext("Working", None, 2.0, state)).kind == "none"

    exit_plan = "Exit plan mode?\nClaude wants to exit plan mode\n❯ Yes"
    assert controller.process(InteractionContext(exit_plan, None, 2.5, state)).kind == "send_keys"
