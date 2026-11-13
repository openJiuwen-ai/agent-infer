# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Classify Claude terminal snapshots into runtime interaction actions."""

import re
from dataclasses import dataclass
from typing import Literal, Protocol

_CONFIRMATION_RETRY_SECONDS = 5.0
_CONFIRMATION_MAX_ATTEMPTS = 2


@dataclass
class InteractionState:
    """Track mutable terminal-interaction evidence for one Claude run."""

    startup_dismissed: bool = False
    yes_confirmation_visible: bool = False
    yes_confirmation_attempts: int = 0
    yes_confirmation_sent_at: float | None = None
    plan_approval_visible: bool = False
    plan_approval_attempts: int = 0
    plan_approval_sent_at: float | None = None
    generic_selection_visible: bool = False
    auto_yes_confirmations: int = 0
    auto_plan_approvals: int = 0
    auto_generic_selections: int = 0
    generic_selection_attempts: int = 0
    generic_selection_sent_at: float | None = None
    confirmation_id: tuple[str | None, str] | None = None


@dataclass
class InteractionContext:
    """Provide transcript and terminal state consumed by interaction handlers."""

    terminal: str
    transcript: list[dict[str, object]] | None
    elapsed_seconds: float
    state: InteractionState


@dataclass
class InteractionAction:
    """Describe the terminal operation selected by an interaction handler."""

    kind: Literal["send_keys", "none", "check_workspace"]
    keys: str = ""


class InteractionHandler(Protocol):
    """Recognize one terminal state and select its interaction action."""

    def recognize(self, ctx: InteractionContext) -> bool: ...

    def act(self, ctx: InteractionContext) -> InteractionAction: ...


_STARTUP_MARKERS = (
    "do you trust",
    "security guide",
    "yes, i trust",
    "quick safety check",
    "let's get started",
    "press enter",
    "choose a theme",
    "theme",
)


class StartupDialogHandler:
    """Dismiss startup dialogs and mark Claude ready at its prompt."""

    def recognize(self, ctx: InteractionContext) -> bool:
        if ctx.state.startup_dismissed:
            return False
        low = ctx.terminal.lower()
        return any(marker in low for marker in _STARTUP_MARKERS) or (
            ("❯" in ctx.terminal or "welcome back" in low) and "[pasted text #" not in low
        )

    def act(self, ctx: InteractionContext) -> InteractionAction:
        low = ctx.terminal.lower()
        if any(marker in low for marker in ("do you trust", "quick safety check", "yes, i trust")):
            options = list(
                re.finditer(
                    r"^\s*([❯>➤▶]?)\s*(?:\d+\.\s*)?(yes, i trust\b.*|no\b.*)$",
                    ctx.terminal,
                    re.IGNORECASE | re.MULTILINE,
                )
            )
            trusted = next(
                (i for i, option in enumerate(options) if option[2].lower().startswith("yes, i trust")), None
            )
            selected = next((i for i, option in enumerate(options) if option[1]), None)
            if trusted is None or selected is None:
                return InteractionAction(kind="none")
            key = "Enter" if selected == trusted else ("Down" if selected < trusted else "Up")
            return InteractionAction(kind="send_keys", keys=key)
        if any(marker in low for marker in _STARTUP_MARKERS):
            return InteractionAction(kind="send_keys", keys="Enter")
        ctx.state.startup_dismissed = True
        return InteractionAction(kind="none")


def _plan_approval_visible(terminal: str) -> bool:
    """Return whether either supported plan approval prompt is visible."""

    text = " ".join(terminal.lower().split())
    return (
        "claude has written up a plan and is ready to execute." in text
        and ("yes, and bypass permissions" in text or "yes, and switch to bypass permissions" in text)
    ) or ("exit plan mode?" in text and "claude wants to exit plan mode" in text and "yes" in text)


class PlanApprovalHandler:
    """Confirm supported plan approvals, retrying once if the prompt persists."""

    def recognize(self, ctx: InteractionContext) -> bool:
        if not _plan_approval_visible(ctx.terminal):
            return False
        state = ctx.state
        # A newly rendered dialog may ignore the first key; retry once after it settles.
        return not state.plan_approval_visible or (
            state.plan_approval_attempts < _CONFIRMATION_MAX_ATTEMPTS
            and state.plan_approval_sent_at is not None
            and ctx.elapsed_seconds - state.plan_approval_sent_at >= _CONFIRMATION_RETRY_SECONDS
        )

    def act(self, ctx: InteractionContext) -> InteractionAction:
        return _approve_plan(ctx)


def _approve_plan(ctx: InteractionContext) -> InteractionAction:
    ctx.state.plan_approval_visible = True
    ctx.state.plan_approval_attempts += 1
    ctx.state.plan_approval_sent_at = ctx.elapsed_seconds
    ctx.state.auto_plan_approvals += 1
    return InteractionAction(kind="send_keys", keys="Enter")


_YES_CONFIRMATION_RE = re.compile(r"(?:❯|>|➤|▶)\s*(?:\d+\.\s*)?Yes\b")

_SELECTION_FOOTER_MARKERS = (
    "enter to select",
    "↑/↓ to navigate",
    "↑ ↓ to move",
    "esc to cancel",
    "space to select",
)

_NUMBERED_SELECTION_RE = re.compile(r"^\s*(?:[❯➤▶]\s*)?\d+\.\s+")
_HIGHLIGHTED_NUMBERED_SELECTION_RE = re.compile(r"^\s*[❯➤▶]\s*\d+\.\s+")

_PROMPT_WINDOW_LINES = 20


def _yes_confirmation_visible(terminal: str) -> bool:
    """Return whether a selectable Yes option is visible near the prompt."""

    lines = [line.strip() for line in terminal.splitlines() if line.strip()]
    return bool(_YES_CONFIRMATION_RE.search("\n".join(lines[-40:])))


class YesConfirmationHandler:
    """Confirm a selected Yes, retrying once if the prompt persists."""

    def recognize(self, ctx: InteractionContext) -> bool:
        if _plan_approval_visible(ctx.terminal) or not _yes_confirmation_visible(ctx.terminal):
            return False
        state = ctx.state
        return not state.yes_confirmation_visible or (
            state.yes_confirmation_attempts < _CONFIRMATION_MAX_ATTEMPTS
            and state.yes_confirmation_sent_at is not None
            and ctx.elapsed_seconds - state.yes_confirmation_sent_at >= _CONFIRMATION_RETRY_SECONDS
        )

    def act(self, ctx: InteractionContext) -> InteractionAction:
        ctx.state.yes_confirmation_visible = True
        ctx.state.yes_confirmation_attempts += 1
        ctx.state.yes_confirmation_sent_at = ctx.elapsed_seconds
        ctx.state.auto_yes_confirmations += 1
        return InteractionAction(kind="send_keys", keys="Enter")


def _generic_selection_visible(terminal: str) -> bool:
    """Return whether an unclaimed numbered multi-choice prompt is visible."""

    low = terminal.lower()
    if not any(marker in low for marker in _SELECTION_FOOTER_MARKERS):
        return False
    lines = terminal.splitlines()[-_PROMPT_WINDOW_LINES:]
    return sum(bool(_NUMBERED_SELECTION_RE.match(line)) for line in lines) >= 2 and any(
        _HIGHLIGHTED_NUMBERED_SELECTION_RE.match(line) for line in lines
    )


class GenericSelectionHandler:
    """Select the highlighted option of any numbered multi-choice prompt once per appearance.

    Defensive fallback for prompts no specific handler claims, such as
    AskUserQuestion whose first option does not begin with Yes:

        ❯ 1. Properties should inherit docstrings
          2. Test needs to be fixed
          3. I need to see more examples

        Enter to select • ↑/↓ to navigate • Esc to cancel

    Selecting the highlighted option is the only non-blocking choice available:
    an autonomous run has no user to ask, and Claude lists its own best guess
    first. A wrong pick surfaces as a failing patch, which the benchmark already
    scores, whereas waiting stalls the run indefinitely -- astropy__astropy-7166
    blocked 78 minutes this way before this handler existed.
    """

    def recognize(self, ctx: InteractionContext) -> bool:
        state = ctx.state
        if state.plan_approval_visible or state.yes_confirmation_visible:
            return False
        if not _generic_selection_visible(ctx.terminal):
            return False
        if not state.generic_selection_visible:
            return True
        return (
            state.generic_selection_attempts < _CONFIRMATION_MAX_ATTEMPTS
            and state.generic_selection_sent_at is not None
            and ctx.elapsed_seconds - state.generic_selection_sent_at >= _CONFIRMATION_RETRY_SECONDS
        )

    def act(self, ctx: InteractionContext) -> InteractionAction:
        state = ctx.state
        state.generic_selection_visible = True
        state.generic_selection_attempts += 1
        state.generic_selection_sent_at = ctx.elapsed_seconds
        state.auto_generic_selections += 1
        return InteractionAction(kind="send_keys", keys="Enter")


class NoProgressHandler:
    """Terminate a single-agent run after sustained idle with patch evidence."""

    _IDLE_TIMEOUT = 10.0

    def __init__(self, enable_idle_detection: bool = True) -> None:
        self._idle_seen_at = 0.0
        self._workspace_checked = False
        self.enable_idle_detection = enable_idle_detection

    def recognize(self, ctx: InteractionContext) -> bool:
        if not self.enable_idle_detection:
            return False
        transcript = ctx.transcript
        if not transcript or len(transcript) < 20:
            self._idle_seen_at = 0.0
            self._workspace_checked = False
            return False
        assistant_count = sum(event.get("type") == "assistant" for event in transcript)
        if assistant_count < 10:
            self._idle_seen_at = 0.0
            self._workspace_checked = False
            return False
        low = ctx.terminal.lower()
        if (
            "❯" in ctx.terminal
            and "[pasted text #" not in low
            and "streaming" not in low
            and "esc to interrupt" not in low
        ):
            if self._idle_seen_at == 0.0:
                self._idle_seen_at = ctx.elapsed_seconds
                return False
            return not self._workspace_checked and ctx.elapsed_seconds - self._idle_seen_at >= self._IDLE_TIMEOUT
        self._idle_seen_at = 0.0
        self._workspace_checked = False
        return False

    def act(self, ctx: InteractionContext) -> InteractionAction:
        self._workspace_checked = True
        return InteractionAction(kind="check_workspace")


class InteractionController:
    """Return the first action recognized by the configured handler chain."""

    def __init__(
        self,
        handler_types: tuple[type[InteractionHandler], ...],
        enable_idle_detection: bool = True,
    ) -> None:
        self._handlers: list[InteractionHandler] = [handler_type() for handler_type in handler_types]
        self._handlers.append(NoProgressHandler(enable_idle_detection=enable_idle_detection))

    def process(self, ctx: InteractionContext) -> InteractionAction:
        visible = (
            ("yes_confirmation", _yes_confirmation_visible(ctx.terminal)),
            ("plan_approval", _plan_approval_visible(ctx.terminal)),
            ("generic_selection", _generic_selection_visible(ctx.terminal)),
        )
        changed = False
        if any(shown for _, shown in visible):
            tool_id, prompt = _confirmation_identity(ctx)
            previous = ctx.state.confirmation_id
            if previous is not None:
                changed = prompt != previous[1] or (
                    tool_id is not None and previous[0] is not None and tool_id != previous[0]
                )
                # Transcript writes may lag behind the terminal, or finish before it redraws.
                if prompt == previous[1] and tool_id is None:
                    tool_id = previous[0]
            ctx.state.confirmation_id = tool_id, prompt
        else:
            ctx.state.confirmation_id = None
        for name, shown in visible:
            if getattr(ctx.state, f"{name}_visible") and (changed or not shown):
                setattr(ctx.state, f"{name}_visible", False)
                setattr(ctx.state, f"{name}_attempts", 0)
                setattr(ctx.state, f"{name}_sent_at", None)
        for handler in self._handlers:
            if handler.recognize(ctx):
                return handler.act(ctx)
        return InteractionAction(kind="none")


def _confirmation_identity(ctx: InteractionContext) -> tuple[str | None, str]:
    """Identify the pending operation and dialog, excluding the changing footer."""
    blocks: list[dict] = []
    for event in ctx.transcript or []:
        message = event.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            blocks.extend(block for block in message["content"] if isinstance(block, dict))
    completed = {
        b["tool_use_id"] for b in blocks if b.get("type") == "tool_result" and isinstance(b.get("tool_use_id"), str)
    }
    tool_id = next(
        (
            b["id"]
            for b in blocks
            if b.get("type") == "tool_use" and isinstance(b.get("id"), str) and b["id"] not in completed
        ),
        None,
    )
    lines = ctx.terminal.splitlines()
    selected = next(
        (
            i
            for i in reversed(range(len(lines)))
            if _YES_CONFIRMATION_RE.search(lines[i]) or _HIGHLIGHTED_NUMBERED_SELECTION_RE.match(lines[i])
        ),
        len(lines),
    )
    prompt = " ".join(
        line.strip()
        for line in lines[max(0, selected - _PROMPT_WINDOW_LINES) : selected + 1]
        if line.strip() and not re.match(r"^[\u2800-\u28ff✢✳✶✻✽]", line.strip())
    )
    return tool_id, " ".join(prompt.split())


def handler_types_for_profile(interaction_profile: str) -> tuple[type[InteractionHandler], ...]:
    """Return terminal handlers selected by a Claude interaction profile."""

    if interaction_profile == "single":
        return (StartupDialogHandler, YesConfirmationHandler, GenericSelectionHandler)
    if interaction_profile == "plan-subagent":
        return (StartupDialogHandler, PlanApprovalHandler, YesConfirmationHandler, GenericSelectionHandler)
    raise ValueError(f"Unsupported Claude interaction profile: {interaction_profile!r}")
