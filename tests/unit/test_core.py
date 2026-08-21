"""Phase 2 core infrastructure tests — CLAUDE.md §1, §8.

The emphasis is on the properties that keep the account alive, not on line coverage:

* a hard constant cannot be moved, from any source;
* the square-off deadline cannot be postponed by moving the wall clock;
* the daily lock fails *safe* when it cannot be read;
* nothing accidentally resolves to LIVE.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import SecretStr

from tachyon.core import config as config_module
from tachyon.core import constants, eventloop
from tachyon.core.clock import (
    IST,
    ManualClock,
    MonotonicDeadline,
    SessionEvent,
    SessionWatchdog,
    is_entry_window_open,
    now_ist,
    to_ist,
)
from tachyon.core.config import ProtectedConstantOverrideError, Settings
from tachyon.core.constants import FrozenConstantError, TradingMode
from tachyon.core.logger import configure_logging, get_logger, reset_logging, shutdown_logging
from tachyon.core.state import (
    DailyLock,
    LockStatus,
    StateMachine,
    StateTransitionError,
    TradingState,
    resolve_boot_state,
)
from tachyon.risk.budget import SessionBudget
from tests.conftest import REAL_SETTINGS_YAML


def _clock_at(hh: int, mm: int, ss: int = 0, mono: float = 1000.0) -> ManualClock:
    """A ManualClock pinned to a given IST wall time on a fixed trading date."""
    return ManualClock(wall=datetime(2026, 8, 10, hh, mm, ss, tzinfo=IST), mono=mono)


# ──────────────────────────────────────────────────────────────────────────────
# constants.py
# ──────────────────────────────────────────────────────────────────────────────


class TestConstants:
    def test_values_match_the_constitution(self) -> None:
        assert constants.AUTO_SQUAREOFF_IST == time(15, 15)
        assert constants.NO_NEW_ENTRIES_IST == time(15, 0)
        assert constants.DAILY_LOSS_LIMIT_INR == Decimal("500")
        assert constants.PER_TRADE_RISK_INR == Decimal("100")
        assert constants.TRADING_MODE is TradingMode.PAPER

    def test_money_is_decimal_never_float(self) -> None:
        assert isinstance(constants.DAILY_LOSS_LIMIT_INR, Decimal)
        assert isinstance(constants.PER_TRADE_RISK_INR, Decimal)

    def test_per_trade_risk_cannot_exceed_the_daily_budget(self) -> None:
        assert constants.PER_TRADE_RISK_INR < constants.DAILY_LOSS_LIMIT_INR

    def test_entry_gate_closes_before_squareoff(self) -> None:
        assert constants.NO_NEW_ENTRIES_IST < constants.AUTO_SQUAREOFF_IST
        assert constants.AUTO_SQUAREOFF_IST < constants.MARKET_CLOSE_IST

    def test_constants_cannot_be_reassigned_at_runtime(self) -> None:
        with pytest.raises(FrozenConstantError):
            constants.DAILY_LOSS_LIMIT_INR = Decimal("5000")  # type: ignore[misc]

    def test_constants_cannot_be_deleted(self) -> None:
        with pytest.raises(FrozenConstantError):
            del constants.AUTO_SQUAREOFF_IST  # type: ignore[misc]

    def test_reassignment_attempt_leaves_the_value_intact(self) -> None:
        with pytest.raises(FrozenConstantError):
            constants.DAILY_LOSS_LIMIT_INR = Decimal("999999")  # type: ignore[misc]
        assert constants.DAILY_LOSS_LIMIT_INR == Decimal("500")


# ──────────────────────────────────────────────────────────────────────────────
# config.py
# ──────────────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_repository_settings_yaml_loads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The shipped config must parse. The only test that reads the operator's real file.

        The suite is otherwise isolated from it (``tests/conftest.py``), so this opts back in
        deliberately — the file *is* the subject here, and a settings.yaml that no longer loads
        would take tomorrow's boot down.
        """
        monkeypatch.setattr(config_module, "SETTINGS_YAML", REAL_SETTINGS_YAML)
        settings = Settings()
        assert settings.session.timezone == "Asia/Kolkata"
        assert settings.watchlist, "the shipped settings.yaml should define a watchlist"

    def test_repository_settings_yaml_resolves_a_sane_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whatever capital the operator has configured must resolve to usable limits.

        Deliberately asserts no specific rupee amount — that is the operator's decision (§1.3).
        It asserts the *shape*: both limits positive, and per-trade never exceeding the day, a
        combination that would otherwise only be discovered by the boot log at 09:15.
        """
        monkeypatch.setattr(config_module, "SETTINGS_YAML", REAL_SETTINGS_YAML)
        capital = Settings().capital
        budget = SessionBudget.resolve(
            capital=capital.session_budget_inr,
            drawdown_pct=capital.max_daily_drawdown_pct,
            per_trade_pct=capital.per_trade_risk_pct,
        )
        assert budget.daily_loss_limit > 0
        assert budget.per_trade_risk > 0
        assert budget.per_trade_risk <= budget.daily_loss_limit

    def test_yaml_override_of_a_hard_constant_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rogue = tmp_path / "settings.yaml"
        rogue.write_text("DAILY_LOSS_LIMIT_INR: 5000\n", encoding="utf-8")
        monkeypatch.setattr(config_module, "SETTINGS_YAML", rogue)

        with pytest.raises(ProtectedConstantOverrideError, match="DAILY_LOSS_LIMIT_INR"):
            Settings()

    def test_nested_yaml_override_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard must reach protected names at any depth, not just the top level."""
        rogue = tmp_path / "settings.yaml"
        rogue.write_text("execution:\n  sl_atr_multiplier: 0.5\n", encoding="utf-8")
        monkeypatch.setattr(config_module, "SETTINGS_YAML", rogue)

        with pytest.raises(ProtectedConstantOverrideError, match="SL_ATR_MULTIPLIER|sl_atr"):
            Settings()

    def test_near_miss_alias_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`per_trade_risk_fraction` is not a real field — it must fail, not be ignored."""
        rogue = tmp_path / "settings.yaml"
        rogue.write_text("sizing:\n  per_trade_risk_fraction: 0.9\n", encoding="utf-8")
        monkeypatch.setattr(config_module, "SETTINGS_YAML", rogue)

        with pytest.raises(ProtectedConstantOverrideError):
            Settings()

    def test_env_override_of_a_hard_constant_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAILY_LOSS_LIMIT_INR", "5000")
        with pytest.raises(ProtectedConstantOverrideError):
            Settings()

    def test_unknown_key_in_a_section_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rogue = tmp_path / "settings.yaml"
        rogue.write_text("ipc:\n  hwm: 100\n  typo_key: 3\n", encoding="utf-8")
        monkeypatch.setattr(config_module, "SETTINGS_YAML", rogue)

        with pytest.raises(Exception, match="typo_key|extra"):
            Settings()

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("LIVE", TradingMode.LIVE),
            ("live", TradingMode.LIVE),
            ("  LiVe  ", TradingMode.LIVE),
            ("PAPER", TradingMode.PAPER),
            ("", TradingMode.PAPER),
            ("nonsense", TradingMode.PAPER),
            ("TRUE", TradingMode.PAPER),
            ("1", TradingMode.PAPER),
        ],
    )
    def test_trading_mode_fails_safe_to_paper(
        self, raw: str, expected: TradingMode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRADING_MODE", raw)
        assert Settings().trading_mode is expected

    def test_trading_mode_defaults_to_paper_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TRADING_MODE", raising=False)
        assert Settings().trading_mode is TradingMode.PAPER

    def test_live_without_credentials_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRADING_MODE", "LIVE")
        monkeypatch.setenv("SMARTAPI_API_KEY", "")
        settings = Settings()
        assert settings.is_live
        with pytest.raises(ValueError, match="SMARTAPI"):
            settings.validate_live_ready()

    def test_secrets_are_not_exposed_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SMARTAPI_PASSWORD", "hunter2-super-secret")
        settings = Settings()
        assert isinstance(settings.smartapi_password, SecretStr)
        assert "hunter2" not in repr(settings)
        assert "hunter2" not in str(settings)

    def test_settings_are_immutable(self) -> None:
        settings = Settings()
        with pytest.raises(Exception, match="frozen|immutable"):
            settings.ui_port = 9999  # type: ignore[misc]

    def test_duplicate_watchlist_symbols_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rogue = tmp_path / "settings.yaml"
        rogue.write_text(
            'watchlist:\n  - {symbol: INFY, token: "1594"}\n  - {symbol: INFY, token: "9999"}\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(config_module, "SETTINGS_YAML", rogue)
        with pytest.raises(Exception, match="Duplicate"):
            Settings()


# ──────────────────────────────────────────────────────────────────────────────
# clock.py
# ──────────────────────────────────────────────────────────────────────────────


class TestClock:
    def test_now_is_always_ist_regardless_of_host_timezone(self) -> None:
        assert now_ist().utcoffset() == IST.utcoffset(datetime(2026, 8, 10))

    def test_naive_datetimes_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="Naive datetime"):
            to_ist(datetime(2026, 8, 10, 10, 0))

    def test_entry_window_boundaries(self) -> None:
        assert not is_entry_window_open(datetime(2026, 8, 10, 9, 19, tzinfo=IST))
        assert is_entry_window_open(datetime(2026, 8, 10, 9, 20, tzinfo=IST))
        assert is_entry_window_open(datetime(2026, 8, 10, 14, 59, 59, tzinfo=IST))
        assert not is_entry_window_open(datetime(2026, 8, 10, 15, 0, tzinfo=IST))

    def test_deadline_counts_down_on_the_monotonic_clock(self) -> None:
        clock = _clock_at(15, 14, 0)
        deadline = MonotonicDeadline.arm("squareoff", time(15, 15), clock=clock)
        assert deadline.remaining(clock) == pytest.approx(60.0)

        clock.advance(59.0)
        assert not deadline.has_elapsed(clock)
        clock.advance(1.0)
        assert deadline.has_elapsed(clock)

    def test_wall_clock_jump_forward_cannot_fire_squareoff_early(self) -> None:
        """An NTP correction that skips ahead must not trigger square-off prematurely."""
        clock = _clock_at(15, 14, 0)
        deadline = MonotonicDeadline.arm("squareoff", time(15, 15), clock=clock)

        clock.jump_wall_clock(3600)  # wall clock now reads 16:14 — monotonic untouched
        assert not deadline.has_elapsed(clock)
        assert deadline.remaining(clock) == pytest.approx(60.0)

    def test_wall_clock_jump_backward_cannot_postpone_squareoff(self) -> None:
        """The dangerous direction: a backward correction must not delay the flatten."""
        clock = _clock_at(15, 14, 0)
        deadline = MonotonicDeadline.arm("squareoff", time(15, 15), clock=clock)

        clock.jump_wall_clock(-3600)  # wall clock now reads 14:14
        clock.advance(60)  # one real minute passes
        assert deadline.has_elapsed(clock)

    def test_deadline_already_past_is_immediately_elapsed(self) -> None:
        """Booting at 15:20 must treat square-off as due now, never as due tomorrow."""
        clock = _clock_at(15, 20, 0)
        deadline = MonotonicDeadline.arm("squareoff", time(15, 15), clock=clock)
        assert deadline.has_elapsed(clock)
        assert deadline.remaining(clock) < 0

    def test_watchdog_fires_events_in_chronological_order(self) -> None:
        clock = _clock_at(9, 0, 0)
        watchdog = SessionWatchdog.arm_for_today(clock, on_date=date(2026, 8, 10))
        assert watchdog.poll() == ()

        clock.advance(60 * 60 * 7)  # 16:00 — every event is now due
        fired = watchdog.poll()
        assert fired == (
            SessionEvent.ENTRIES_OPEN,
            SessionEvent.NO_NEW_ENTRIES,
            SessionEvent.SQUARE_OFF,
            SessionEvent.MARKET_CLOSE,
        )

    def test_watchdog_events_latch_and_report_once(self) -> None:
        clock = _clock_at(15, 14, 30)
        watchdog = SessionWatchdog.arm_for_today(clock, on_date=date(2026, 8, 10))
        watchdog.poll()  # drain ENTRIES_OPEN / NO_NEW_ENTRIES

        clock.advance(31)
        assert SessionEvent.SQUARE_OFF in watchdog.poll()
        assert watchdog.has_fired(SessionEvent.SQUARE_OFF)
        assert SessionEvent.SQUARE_OFF not in watchdog.poll()

    def test_watchdog_latch_survives_a_backward_clock_jump(self) -> None:
        clock = _clock_at(15, 14, 30)
        watchdog = SessionWatchdog.arm_for_today(clock, on_date=date(2026, 8, 10))
        clock.advance(31)
        watchdog.poll()
        assert watchdog.has_fired(SessionEvent.SQUARE_OFF)

        clock.jump_wall_clock(-7200)
        assert watchdog.has_fired(SessionEvent.SQUARE_OFF), "square-off must never un-fire"

    def test_next_poll_delay_never_overshoots_a_deadline(self) -> None:
        clock = _clock_at(15, 14, 59, mono=1000.0)
        watchdog = SessionWatchdog.arm_for_today(clock, on_date=date(2026, 8, 10))
        watchdog.poll()
        assert watchdog.next_poll_delay(tick=0.25) == pytest.approx(0.25)

        clock.advance(0.9)  # 0.1s to square-off, shorter than one tick
        assert watchdog.next_poll_delay(tick=0.25) == pytest.approx(0.1, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# logger.py
# ──────────────────────────────────────────────────────────────────────────────


class TestLogger:
    @pytest.fixture(autouse=True)
    def _clean_logging(self) -> object:
        reset_logging()
        yield
        reset_logging()

    def _read_events(self, log_dir: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in self._read_lines(log_dir)]

    def _read_lines(self, log_dir: Path) -> list[str]:
        # The file sink runs on a QueueListener thread, so a read taken immediately after
        # .info() races the writer. shutdown_logging() drains the queue first — without it
        # these assertions pass or fail on scheduler luck.
        shutdown_logging()
        files = list(log_dir.glob("*.jsonl"))
        assert files, "expected a JSON log file to be created"
        return [line for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_writes_parseable_json_lines_with_ist_timestamps(self, tmp_path: Path) -> None:
        configure_logging(role="brain", log_dir=tmp_path, pretty_console=False)
        get_logger("test").info("order.placed", symbol="INFY", qty=7)

        events = self._read_events(tmp_path)
        assert len(events) == 1
        event = events[0]
        assert event["event"] == "order.placed"
        assert event["symbol"] == "INFY"
        assert event["role"] == "brain"
        assert event["mode"] == "PAPER"
        assert "ts_mono" in event
        assert "+05:30" in str(event["ts_ist"])

    def test_secrets_are_redacted(self, tmp_path: Path) -> None:
        configure_logging(role="brain", log_dir=tmp_path, pretty_console=False)
        get_logger("test").info(
            "login",
            smartapi_password="hunter2",
            gemini_api_key="AIzaSyTOPSECRET",
            totp="123456",
            nested={"access_token": "jwt-value", "symbol": "INFY"},
        )

        raw = "\n".join(self._read_lines(tmp_path))
        for secret in ("hunter2", "AIzaSyTOPSECRET", "123456", "jwt-value"):
            assert secret not in raw, f"{secret!r} leaked into the log file"
        assert "INFY" in raw, "non-secret context must survive redaction"

    def test_instrument_tokens_are_not_redacted(self, tmp_path: Path) -> None:
        """`token` is an instrument id, not a credential — redacting it would blind the feed."""
        configure_logging(role="ingestor", log_dir=tmp_path, pretty_console=False)
        get_logger("test").info("tick", symbol="INFY", token="1594")

        assert "1594" in "\n".join(self._read_lines(tmp_path))

    def test_json_keys_are_sorted_for_deterministic_diffs(self, tmp_path: Path) -> None:
        configure_logging(role="brain", log_dir=tmp_path, pretty_console=False)
        get_logger("test").info("evt", zulu=1, alpha=2)

        keys = list(json.loads(self._read_lines(tmp_path)[0]).keys())
        assert keys == sorted(keys)


# ──────────────────────────────────────────────────────────────────────────────
# state.py
# ──────────────────────────────────────────────────────────────────────────────


class TestTradingStateMachine:
    def test_normal_session_progression(self) -> None:
        machine = StateMachine(TradingState.BOOTING)
        for target in (
            TradingState.PRE_MARKET,
            TradingState.ACTIVE,
            TradingState.NO_NEW_ENTRIES,
            TradingState.SQUARING_OFF,
            TradingState.SQUARED_OFF,
        ):
            machine.transition_to(target, reason="test")
        assert machine.state is TradingState.SQUARED_OFF

    def test_cannot_reopen_entries_after_the_gate_closes(self) -> None:
        machine = StateMachine(TradingState.NO_NEW_ENTRIES)
        with pytest.raises(StateTransitionError):
            machine.transition_to(TradingState.ACTIVE, reason="sneaky")

    def test_locked_is_absorbing(self) -> None:
        machine = StateMachine(TradingState.ACTIVE)
        machine.lock_out(reason="daily loss limit")
        assert machine.state is TradingState.LOCKED
        assert machine.is_terminal()

        for target in TradingState:
            if target is TradingState.LOCKED:
                continue
            with pytest.raises(StateTransitionError):
                machine.transition_to(target, reason="escape attempt")

    def test_lock_out_is_reachable_from_every_state(self) -> None:
        for start in TradingState:
            if start is TradingState.LOCKED:
                continue
            machine = StateMachine(start)
            machine.lock_out(reason="kill switch")
            assert machine.state is TradingState.LOCKED

    def test_repeated_lock_out_does_not_raise(self) -> None:
        """A second breach reporting in must not crash the process doing the shutting down."""
        machine = StateMachine(TradingState.ACTIVE)
        machine.lock_out(reason="first")
        machine.lock_out(reason="second")
        assert machine.state is TradingState.LOCKED

    def test_entries_only_permitted_while_active(self) -> None:
        for state in TradingState:
            machine = StateMachine(state)
            assert machine.may_open_position() is (state is TradingState.ACTIVE)

    def test_exits_are_permitted_in_every_state(self) -> None:
        for state in TradingState:
            assert StateMachine(state).may_exit_position() is True

    def test_a_failing_listener_cannot_block_a_transition(self) -> None:
        machine = StateMachine(TradingState.ACTIVE)
        machine.add_listener(lambda _t: (_ for _ in ()).throw(RuntimeError("listener boom")))
        machine.lock_out(reason="kill switch")
        assert machine.state is TradingState.LOCKED


class TestDailyLock:
    def test_absent_lock_permits_trading(self, tmp_path: Path) -> None:
        lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=_clock_at(10, 0))
        assert lock.status() is LockStatus.ABSENT
        assert not lock.is_engaged()

    def test_engaging_latches_for_today(self, tmp_path: Path) -> None:
        clock = _clock_at(11, 30)
        lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)
        lock.engage("DAILY_LOSS_LIMIT", realised_pnl_inr=Decimal("-500.25"))

        assert lock.status() is LockStatus.ENGAGED
        assert lock.is_engaged()
        record = lock.read()
        assert record is not None
        assert record.reason == "DAILY_LOSS_LIMIT"
        assert record.realised_pnl_inr == Decimal("-500.25")
        assert record.lock_date == date(2026, 8, 10)

    def test_first_line_is_the_iso_date(self, tmp_path: Path) -> None:
        path = tmp_path / "daily_lock.txt"
        DailyLock(path=path, clock=_clock_at(11, 30)).engage("TEST")
        assert path.read_text(encoding="utf-8").splitlines()[0] == "2026-08-10"

    def test_yesterdays_lock_is_stale_and_does_not_block_today(self, tmp_path: Path) -> None:
        path = tmp_path / "daily_lock.txt"
        path.write_text("2026-08-09\n{}\n", encoding="utf-8")
        lock = DailyLock(path=path, clock=_clock_at(9, 30))

        assert lock.status() is LockStatus.STALE
        assert not lock.is_engaged()

    def test_stale_lock_is_archived_rather_than_overwritten(self, tmp_path: Path) -> None:
        path = tmp_path / "daily_lock.txt"
        path.write_text("2026-08-09\n{}\n", encoding="utf-8")
        DailyLock(path=path, clock=_clock_at(11, 0)).engage("NEW_BREACH")

        assert (tmp_path / "daily_lock.2026-08-09.txt").is_file(), "history must be preserved"
        assert path.read_text(encoding="utf-8").splitlines()[0] == "2026-08-10"

    @pytest.mark.parametrize(
        "corrupt", ["", "not-a-date\n", "\x00\x01garbage", "2026-13-45\n", "   \n"]
    )
    def test_unreadable_lock_fails_safe_to_engaged(self, tmp_path: Path, corrupt: str) -> None:
        """If we cannot prove trading is permitted, it is not permitted."""
        path = tmp_path / "daily_lock.txt"
        path.write_text(corrupt, encoding="utf-8")
        lock = DailyLock(path=path, clock=_clock_at(10, 0))

        assert lock.status() is LockStatus.UNREADABLE
        assert lock.is_engaged(), "a lock we cannot parse must still lock us out"

    def test_engage_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "daily_lock.txt"
        DailyLock(path=path, clock=_clock_at(11, 0)).engage("TEST")
        assert path.is_file()


class TestBootResolution:
    @pytest.mark.parametrize(
        ("hh", "mm", "expected"),
        [
            (8, 45, TradingState.PRE_MARKET),
            (9, 14, TradingState.PRE_MARKET),
            (9, 15, TradingState.ACTIVE),
            (12, 0, TradingState.ACTIVE),
            (14, 59, TradingState.ACTIVE),
            (15, 0, TradingState.NO_NEW_ENTRIES),
            (15, 14, TradingState.NO_NEW_ENTRIES),
            (15, 15, TradingState.SQUARING_OFF),
            (15, 29, TradingState.SQUARING_OFF),
            (15, 30, TradingState.SQUARED_OFF),
            (18, 0, TradingState.SQUARED_OFF),
        ],
    )
    def test_state_follows_the_ist_wall_clock(
        self, tmp_path: Path, hh: int, mm: int, expected: TradingState
    ) -> None:
        clock = _clock_at(hh, mm)
        lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)
        assert resolve_boot_state(lock, clock) is expected

    def test_restart_between_squareoff_and_close_re_runs_the_flatten(self, tmp_path: Path) -> None:
        """15:20 must resolve to SQUARING_OFF — we cannot assume the dead process finished."""
        clock = _clock_at(15, 20)
        lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)
        assert resolve_boot_state(lock, clock) is TradingState.SQUARING_OFF

    def test_engaged_lock_overrides_the_clock_entirely(self, tmp_path: Path) -> None:
        clock = _clock_at(10, 0)  # mid-session, would otherwise be ACTIVE
        lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)
        lock.engage("DAILY_LOSS_LIMIT", realised_pnl_inr=Decimal("-500"))

        assert resolve_boot_state(lock, clock) is TradingState.LOCKED

    def test_restart_after_a_breach_cannot_resume_trading(self, tmp_path: Path) -> None:
        """The whole point of the on-disk latch: restarting is not a way back in."""
        clock = _clock_at(10, 0)
        lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)
        lock.engage("DAILY_LOSS_LIMIT", realised_pnl_inr=Decimal("-501"))

        machine = StateMachine.boot(lock=lock, clock=clock)
        assert machine.state is TradingState.LOCKED
        assert not machine.may_open_position()

    def test_clean_boot_reaches_active_mid_session(self, tmp_path: Path) -> None:
        clock = _clock_at(10, 0)
        lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)
        machine = StateMachine.boot(lock=lock, clock=clock)

        assert machine.state is TradingState.ACTIVE
        assert machine.may_open_position()


# ──────────────────────────────────────────────────────────────────────────────
# eventloop.py
# ──────────────────────────────────────────────────────────────────────────────


class TestEventLoop:
    """CLAUDE.md §2.2 — the loop must be able to host a zmq.asyncio socket.

    On Windows this is not a performance question. The default Proactor loop lacks
    ``add_reader``, so the Brain would consume *zero* ticks while looking perfectly alive.
    """

    def test_the_chosen_loop_supports_zmq(self) -> None:
        loop = eventloop.new_event_loop()
        try:
            eventloop.assert_zmq_compatible(loop)
            assert hasattr(loop, "add_reader")
        finally:
            loop.close()

    def test_windows_never_returns_the_proactor_loop(self) -> None:
        if sys.platform != "win32":
            pytest.skip("Windows-specific failure mode")
        loop = eventloop.new_event_loop()
        try:
            assert "Proactor" not in type(loop).__name__
        finally:
            loop.close()

    def test_an_incompatible_loop_is_refused_with_a_legible_message(self) -> None:
        class NoReader:
            pass

        with pytest.raises(eventloop.EventLoopUnsuitableError) as exc:
            eventloop.assert_zmq_compatible(NoReader())  # type: ignore[arg-type]
        assert "add_reader" in str(exc.value)
        assert eventloop.ACCELERATOR in str(exc.value)

    def test_run_executes_a_coroutine_and_returns_its_value(self) -> None:
        async def work() -> int:
            await asyncio.sleep(0)
            return 42

        assert eventloop.run(work()) == 42

    def test_run_propagates_exceptions(self) -> None:
        async def boom() -> None:
            raise ValueError("propagated")

        with pytest.raises(ValueError, match="propagated"):
            eventloop.run(boom())
