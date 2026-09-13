"""confirm_live headless safety — Vector 5 regression tests.

The live incident: Task Scheduler boot with TRADING_MODE=LIVE hit the confirmation
prompt with no console attached. ``input()`` raised ``OSError: [Errno 9] Bad file
descriptor``, which the old handler did not catch, and the process died mid-boot.

Contract after the fix:
- non-LIVE modes pass unconditionally;
- ``TACHYON_ASSUME_YES=1`` / ``--yes`` handshake confirms loudly;
- no TTY ⇒ refuse (never prompt, never crash);
- a real TTY still gets asked, and anything other than ``LIVE`` aborts.
"""

from __future__ import annotations

import sys

import pytest

from tachyon.core.config import Settings
from tachyon.core.constants import TradingMode
from tachyon.main import confirm_live


def _live_settings() -> Settings:
    return Settings(trading_mode=TradingMode.LIVE)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TACHYON_ASSUME_YES", raising=False)
    monkeypatch.delenv("TACHYON_NON_INTERACTIVE", raising=False)


class _TtyStdin:
    @staticmethod
    def isatty() -> bool:
        return True


class _NoConsole:
    @staticmethod
    def isatty() -> bool:
        return False


def _raise_eof(*_a: object, **_k: object) -> str:
    raise EOFError


def _raise_errno9(*_a: object, **_k: object) -> str:
    raise OSError(9, "Bad file descriptor")


def test_paper_mode_never_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", None)  # worst case: no stdin at all
    assert confirm_live(Settings(trading_mode=TradingMode.PAPER)) is True


def test_env_handshake_confirms_headless_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", None)  # would crash input()
    monkeypatch.setenv("TACHYON_ASSUME_YES", "1")
    assert confirm_live(_live_settings()) is True


def test_assume_yes_kwarg_confirms_headless_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", None)
    assert confirm_live(_live_settings(), assume_yes=True) is True


def test_headless_without_handshake_refuses_not_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Errno 9 regression: no console must refuse cleanly, never raise."""
    monkeypatch.setattr(sys, "stdin", None)
    monkeypatch.setattr("builtins.input", _raise_errno9)
    # Refusal happens at the TTY gate BEFORE input() would ever be touched.
    assert confirm_live(_live_settings()) is False


def test_lying_tty_with_broken_console_still_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A console claiming isatty()=True whose read raises Errno 9 is caught too."""

    class _LyingConsole:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", _LyingConsole())
    monkeypatch.setattr("builtins.input", _raise_errno9)
    assert confirm_live(_live_settings()) is False


def test_interactive_tty_requires_exact_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _TtyStdin())
    monkeypatch.setattr("builtins.input", lambda *_a: "live")  # wrong word on purpose
    assert confirm_live(_live_settings()) is False

    monkeypatch.setattr("builtins.input", lambda *_a: "LIVE")
    assert confirm_live(_live_settings()) is True


def test_tty_prompt_rejects_when_input_raises_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _TtyStdin())
    monkeypatch.setattr("builtins.input", _raise_eof)
    assert confirm_live(_live_settings()) is False


def test_assume_yes_env_garbage_falls_through_to_headless_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TACHYON_ASSUME_YES", "maybe?")
    monkeypatch.setattr(sys, "stdin", _NoConsole())
    # Not "1/true/yes" ⇒ handshake does not apply ⇒ headless refusal path.
    assert confirm_live(_live_settings()) is False
