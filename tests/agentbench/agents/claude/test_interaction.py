# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Phase 1 boundary tests for interaction handlers and controller."""

from __future__ import annotations

import pytest

from agentinfer.agentbench.agents.claude.interaction import (
    GenericSelectionHandler,
    InteractionContext,
    InteractionController,
    InteractionState,
    NoProgressHandler,
    PlanApprovalHandler,
    StartupDialogHandler,
    YesConfirmationHandler,
    handler_types_for_profile,
)

_ASK_USER_QUESTION_TERMINAL = """
Does the failing test expect properties to inherit docstrings?

❯ 1. Properties should inherit docstrings
  2. Test needs to be fixed
  3. I need to see more examples

Enter to select • ↑/↓ to navigate • Esc to cancel
"""


def _selection_controller() -> InteractionController:
    return InteractionController(
        handler_types_for_profile("plan-subagent"),
        enable_idle_detection=False,
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
    ctx = InteractionContext("Security Guide — press Enter to continue", None, 0.0, state)
    action = handler.act(ctx)
    assert action.kind == "send_keys"
    assert action.keys == "Enter"


@pytest.mark.parametrize(
    ("options", "selected_yes", "key"),
    [
        ("❯ No, exit\n  Yes, I trust this folder", "  No, exit\n❯ Yes, I trust this folder", "Down"),
        ("  Yes, I trust this folder\n❯ No, exit", "❯ Yes, I trust this folder\n  No, exit", "Up"),
        ("  1. Yes, I trust this folder\n❯ 2. No, exit", "❯ 1. Yes, I trust this folder\n  2. No, exit", "Up"),
    ],
)
def test_startup_selects_trusted_option_before_confirming(options, selected_yes, key) -> None:
    controller = InteractionController(handler_types_for_profile("single"))
    state = InteractionState()
    title = "Accessing workspace:\n/tmp/task\nQuick safety check: Is this a project you trust?\n"
    action = controller.process(InteractionContext(title + options, None, 0.0, state))
    assert (action.kind, action.keys) == ("send_keys", key)
    assert state.startup_dismissed is False
    action = controller.process(InteractionContext(title + selected_yes, None, 1.0, state))
    assert (action.kind, action.keys) == ("send_keys", "Enter")
    controller.process(InteractionContext("Welcome back\n❯ ", None, 2.0, state))
    assert state.startup_dismissed is True


@pytest.mark.parametrize(
    "terminal",
    ["Do you trust?", "Quick safety check\n❯ No, exit", "Yes, I trust this folder\nNo, exit"],
)
def test_startup_does_not_confirm_unrecognized_trust_selection(terminal: str) -> None:
    state = InteractionState()
    action = StartupDialogHandler().act(InteractionContext(terminal, None, 0.0, state))
    assert action.kind == "none"
    assert state.startup_dismissed is False


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


@pytest.mark.parametrize("profile", ["single", "plan-subagent"])
def test_edit_confirmation_retries_once_and_resets_after_prompt_clears(profile: str) -> None:
    controller = InteractionController(handler_types_for_profile(profile), enable_idle_detection=profile == "single")
    state = InteractionState(startup_dismissed=True)
    terminal = (
        "Do you want to make this edit to separable.py?\n"
        "❯ 1. Yes\n  2. Yes, and switch to accept edits\n  3. No\nEsc to cancel · Tab to amend"
    )

    assert controller.process(InteractionContext(terminal, None, 10.0, state)).keys == "Enter"
    assert controller.process(InteractionContext(terminal, None, 14.9, state)).kind == "none"
    assert controller.process(InteractionContext(terminal, None, 15.0, state)).keys == "Enter"
    assert controller.process(InteractionContext(terminal, None, 20.0, state)).kind == "none"
    assert controller.process(InteractionContext(terminal, None, 300.0, state)).kind == "none"
    assert state.auto_yes_confirmations == 2
    assert state.auto_generic_selections == 0
    controller.process(InteractionContext("Working", None, 301.0, state))
    assert state.yes_confirmation_attempts == 0
    assert state.yes_confirmation_sent_at is None
    assert controller.process(InteractionContext(terminal, None, 302.0, state)).keys == "Enter"


def test_yes_confirmation_does_not_retry_when_selection_changes_to_no() -> None:
    controller = InteractionController((YesConfirmationHandler,), enable_idle_detection=False)
    state = InteractionState(startup_dismissed=True)
    yes = "Do you want to make this edit?\n❯ 1. Yes\n  2. No"
    no = "Do you want to make this edit?\n  1. Yes\n❯ 2. No"
    assert controller.process(InteractionContext(yes, None, 0.0, state)).keys == "Enter"
    assert controller.process(InteractionContext(no, None, 5.0, state)).kind == "none"
    assert state.yes_confirmation_attempts == 0
    assert state.auto_yes_confirmations == 1


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


def test_no_progress_handler_resets_idle_timer_while_claude_is_busy() -> None:
    handler = NoProgressHandler()
    state = InteractionState()
    transcript: list[dict[str, object]] = [{"type": "assistant"} for _ in range(20)]
    busy = "✽ Working… (25s)\n❯\n⏵⏵ accept edits on · esc to interrupt"

    assert handler.recognize(InteractionContext("❯", transcript, 100.0, state)) is False
    assert handler.recognize(InteractionContext(busy, transcript, 110.0, state)) is False
    assert handler.recognize(InteractionContext(busy, transcript, 140.0, state)) is False
    assert handler.recognize(InteractionContext("❯", transcript, 150.0, state)) is False
    assert handler.recognize(InteractionContext("❯", transcript, 159.9, state)) is False
    assert handler.recognize(InteractionContext("❯", transcript, 160.0, state)) is True


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


@pytest.mark.parametrize(
    "option",
    ["Yes, and bypass permissions", "Yes, and switch to BYPASS PERMISSIONS (no further prompts)"],
)
def test_plan_approval_handles_wrapping_and_does_not_depend_on_generic_yes_state(option: str) -> None:
    controller = InteractionController(handler_types_for_profile("plan-subagent"), enable_idle_detection=False)
    state = InteractionState(startup_dismissed=True, yes_confirmation_visible=True)
    terminal = (
        "Ready to code?\nClaude has written up a plan and is ready to\nexecute. Would you like to proceed?\n"
        f"❯ 1. {option}\n  2. Yes, manually approve edits\n  3. Tell Claude what to change"
    )
    action = controller.process(InteractionContext(terminal, None, 1.0, state))
    assert (action.kind, action.keys) == ("send_keys", "Enter")
    assert state.auto_plan_approvals == 1
    assert state.auto_yes_confirmations == 0
    assert controller.process(InteractionContext(terminal, None, 1.5, state)).kind == "none"


def test_plan_approval_retries_once_and_resets_after_prompt_clears() -> None:
    controller = InteractionController(handler_types_for_profile("plan-subagent"), enable_idle_detection=False)
    state = InteractionState(startup_dismissed=True)
    terminal = "Claude has written up a plan and is ready to execute.\n❯ 1. Yes, and switch to BYPASS PERMISSIONS"

    assert controller.process(InteractionContext(terminal, None, 10.0, state)).keys == "Enter"
    assert controller.process(InteractionContext(terminal, None, 14.9, state)).kind == "none"
    assert controller.process(InteractionContext(terminal, None, 15.0, state)).keys == "Enter"
    assert controller.process(InteractionContext(terminal, None, 20.0, state)).kind == "none"
    assert state.auto_plan_approvals == 2
    controller.process(InteractionContext("Working", None, 21.0, state))
    assert state.plan_approval_attempts == 0
    assert controller.process(InteractionContext(terminal, None, 22.0, state)).keys == "Enter"
    assert state.plan_approval_attempts == 1


def test_generic_selection_handles_non_yes_ask_user_question() -> None:
    state = InteractionState(startup_dismissed=True)
    action = _selection_controller().process(InteractionContext(_ASK_USER_QUESTION_TERMINAL, None, 10.0, state))

    assert action.kind == "send_keys"
    assert action.keys == "Enter"
    assert state.auto_generic_selections == 1


def test_generic_selection_requires_highlighted_numbered_option() -> None:
    terminal = """
> 1. Quoted first option
> 2. Quoted second option

Enter to select
"""
    ctx = InteractionContext(terminal, None, 10.0, InteractionState(startup_dismissed=True))

    assert GenericSelectionHandler().recognize(ctx) is False


def test_generic_selection_retries_once_after_delay_then_stops() -> None:
    state = InteractionState(startup_dismissed=True)
    controller = _selection_controller()

    assert controller.process(InteractionContext(_ASK_USER_QUESTION_TERMINAL, None, 10.0, state)).kind == "send_keys"
    assert controller.process(InteractionContext(_ASK_USER_QUESTION_TERMINAL, None, 14.9, state)).kind == "none"
    assert controller.process(InteractionContext(_ASK_USER_QUESTION_TERMINAL, None, 15.0, state)).kind == "send_keys"
    assert controller.process(InteractionContext(_ASK_USER_QUESTION_TERMINAL, None, 20.0, state)).kind == "none"
    assert state.auto_generic_selections == 2


def test_generic_selection_retry_budget_resets_after_prompt_clears() -> None:
    state = InteractionState(startup_dismissed=True)
    controller = _selection_controller()

    assert controller.process(InteractionContext(_ASK_USER_QUESTION_TERMINAL, None, 10.0, state)).kind == "send_keys"
    assert controller.process(InteractionContext(_ASK_USER_QUESTION_TERMINAL, None, 15.0, state)).kind == "send_keys"
    assert controller.process(InteractionContext("Working", None, 16.0, state)).kind == "none"
    assert controller.process(InteractionContext(_ASK_USER_QUESTION_TERMINAL, None, 20.0, state)).kind == "send_keys"
    assert state.auto_generic_selections == 3


def test_generic_selection_does_not_claim_yes_prompt() -> None:
    terminal = _ASK_USER_QUESTION_TERMINAL.replace("Properties should inherit docstrings", "Yes, this looks right", 1)
    state = InteractionState(startup_dismissed=True)
    action = _selection_controller().process(InteractionContext(terminal, None, 10.0, state))

    assert action.kind == "send_keys"
    assert state.auto_yes_confirmations == 1
    assert state.auto_generic_selections == 0


@pytest.mark.parametrize(
    "terminal",
    [
        "Do you want to make this edit to a.py?\n❯ 1. Yes\n  2. No\nEsc to cancel",
        "Plan A\nClaude has written up a plan and is ready to execute.\n❯ 1. Yes, and bypass permissions",
        "Question A\n❯ 1. Option A\n  2. Option B\nEnter to select",
    ],
)
def test_new_confirmation_without_blank_frame_gets_its_own_retries(terminal: str) -> None:
    controller = _selection_controller()
    state = InteractionState(startup_dismissed=True)
    for elapsed in (0, 5):
        assert controller.process(InteractionContext(terminal, None, elapsed, state)).keys == "Enter"
    changed = terminal.replace("a.py", "b.py").replace("Plan A", "Plan B").replace("Question A", "Question B")
    assert controller.process(InteractionContext(changed, None, 6, state)).keys == "Enter"
    assert controller.process(InteractionContext(changed, None, 10.9, state)).kind == "none"
    assert controller.process(InteractionContext(changed, None, 11, state)).keys == "Enter"
    assert controller.process(InteractionContext(changed, None, 60, state)).kind == "none"


@pytest.mark.parametrize("profile", ["single", "plan-subagent"])
def test_consecutive_tool_calls_with_same_dialog_have_separate_retries(profile: str) -> None:
    controller = InteractionController(handler_types_for_profile(profile), enable_idle_detection=False)
    state = InteractionState(startup_dismissed=True)
    terminal = "Do you want to make this edit to a.py?\n❯ 1. Yes\n  2. No"
    first: dict[str, object] = {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "id": "edit-1", "name": "Edit"}]},
    }
    transcript: list[dict[str, object]] = [first]
    for elapsed in (0, 5):
        assert controller.process(InteractionContext(terminal, transcript, elapsed, state)).keys == "Enter"
    transcript += [
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "edit-1"}]}},
        first,  # Claude may repeat an assistant tool-use record after its result.
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "edit-2", "name": "Edit"}]}},
    ]
    assert controller.process(InteractionContext(terminal, transcript, 6, state)).keys == "Enter"
    assert controller.process(InteractionContext(terminal + "\n⠋ Working (10s)", transcript, 7, state)).kind == "none"


def test_transcript_arrival_and_tool_result_do_not_reset_same_dialog() -> None:
    controller = _selection_controller()
    state = InteractionState(startup_dismissed=True)
    terminal = "Do you want to make this edit to a.py?\n❯ 1. Yes\n  2. No"
    assert controller.process(InteractionContext(terminal, None, 0, state)).keys == "Enter"
    transcript: list[dict[str, object]] = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "edit-1"}]}}
    ]
    assert controller.process(InteractionContext(terminal, transcript, 1, state)).kind == "none"
    assert controller.process(InteractionContext(terminal, transcript, 5, state)).keys == "Enter"
    transcript.append({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "edit-1"}]}})
    assert controller.process(InteractionContext(terminal, transcript, 6, state)).kind == "none"
