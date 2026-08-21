"""Phase 6 risk engine tests — CLAUDE.md §1, §4.

The emphasis is on **fail-safe**: what the gate does when a check is broken, when the data is
undefined, when the disk refuses a write. Anything ambiguous must resolve to a veto. A test
that only proves the happy path proves nothing about a component whose job is to say no.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tachyon.core.clock import IST, ManualClock, SessionEvent
from tachyon.core.constants import DAILY_LOSS_LIMIT_INR, TradingMode
from tachyon.core.state import DailyLock, StateMachine, TradingState
from tachyon.ipc.monitor import FeedMonitor
from tachyon.math_engine import warmup
from tachyon.math_engine.warmup import reset_warm_state
from tachyon.risk.engine import RiskDecision, RiskEngine, VetoReason
from tachyon.risk.tracker import PnLTracker, PositionRegistry
from tachyon.risk.watchdog import SquareOffWatchdog

TRADING_DAY = datetime(2026, 8, 10, 11, 0, tzinfo=IST)


@pytest.fixture(scope="session", autouse=True)
def _warm_engine() -> None:
    assert warmup() is True


def _clock(hh: int = 11, mm: int = 0, ss: int = 0, mono: float = 1000.0) -> ManualClock:
    return ManualClock(wall=datetime(2026, 8, 10, hh, mm, ss, tzinfo=IST), mono=mono)


class _Harness:
    """A fully wired risk stack whose every input can be perturbed."""

    def __init__(self, tmp_path: Path, clock: ManualClock) -> None:
        self.clock = clock
        self.machine = StateMachine(TradingState.ACTIVE, clock=clock)
        self.lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)
        self.pnl = PnLTracker(self.machine, daily_lock=self.lock, clock=clock)
        self.monitor = FeedMonitor(clock=clock)
        self.positions = PositionRegistry()
        self.engine = RiskEngine(self.machine, self.pnl, self.monitor, self.positions, clock=clock)
        self.monitor.record()  # feed is alive by default


@pytest.fixture
def harness(tmp_path: Path) -> _Harness:
    return _Harness(tmp_path, _clock())


# ──────────────────────────────────────────────────────────────────────────────
# The gate — happy path and each veto in isolation
# ──────────────────────────────────────────────────────────────────────────────


class TestVetoGate:
    def test_allows_when_everything_is_healthy(self, harness: _Harness) -> None:
        decision = harness.engine.evaluate("RELIANCE")
        assert decision.allowed
        assert decision.reason is None
        assert harness.engine.can_open_position("RELIANCE")

    def test_decision_is_truthy_when_allowed(self, harness: _Harness) -> None:
        assert bool(harness.engine.evaluate("RELIANCE"))
        harness.machine.lock_out("test")
        assert not bool(harness.engine.evaluate("RELIANCE"))

    @pytest.mark.parametrize(
        "state",
        [s for s in TradingState if s is not TradingState.ACTIVE],
    )
    def test_vetoes_in_every_non_active_state(self, tmp_path: Path, state: TradingState) -> None:
        harness = _Harness(tmp_path, _clock())
        harness.machine = StateMachine(state, clock=harness.clock)
        harness.engine = RiskEngine(
            harness.machine,
            harness.pnl,
            harness.monitor,
            harness.positions,
            clock=harness.clock,
        )
        decision = harness.engine.evaluate("RELIANCE")
        assert not decision.allowed
        assert decision.reason is VetoReason.STATE_NOT_ACTIVE

    def test_vetoes_when_daily_lock_engaged(self, harness: _Harness) -> None:
        harness.lock.engage("DAILY_LOSS_LIMIT", realised_pnl_inr=Decimal("-500"))
        decision = harness.engine.evaluate("RELIANCE")
        assert decision.reason is VetoReason.DAILY_LOCK_ENGAGED

    @pytest.mark.parametrize(
        ("hh", "mm"),
        [(9, 0), (9, 19), (15, 0), (15, 1), (15, 30), (18, 0)],
    )
    def test_vetoes_outside_the_entry_window(self, tmp_path: Path, hh: int, mm: int) -> None:
        harness = _Harness(tmp_path, _clock(hh, mm))
        decision = harness.engine.evaluate("RELIANCE")
        assert decision.reason is VetoReason.OUTSIDE_ENTRY_WINDOW

    def test_allows_at_the_window_boundaries(self, tmp_path: Path) -> None:
        opening = _Harness(tmp_path, _clock(9, 20))
        assert opening.engine.evaluate("RELIANCE").allowed

        closing = _Harness(tmp_path, _clock(14, 59, 59))
        assert closing.engine.evaluate("RELIANCE").allowed

    def test_vetoes_when_feed_is_stale(self, harness: _Harness) -> None:
        harness.clock.advance(2.5)
        decision = harness.engine.evaluate("RELIANCE")
        assert decision.reason is VetoReason.FEED_STALE
        assert "2.5" in decision.detail

    def test_vetoes_when_loss_limit_breached(self, harness: _Harness) -> None:
        """Booking the limit blocks entries.

        The reported reason is ``STATE_NOT_ACTIVE`` rather than ``LOSS_LIMIT_BREACHED``,
        because tripping the switch also locks the state machine and the state check runs
        earlier. Two independent mechanisms both refuse the trade, which is the intent.
        """
        harness.pnl.book_realised(Decimal("-500"))
        decision = harness.engine.evaluate("RELIANCE")
        assert not decision.allowed
        assert decision.reason in {
            VetoReason.LOSS_LIMIT_BREACHED,
            VetoReason.STATE_NOT_ACTIVE,
        }

    def test_loss_limit_check_vetoes_on_its_own(self, tmp_path: Path) -> None:
        """Isolate check 6: a breached tracker blocks entries even if the state says ACTIVE.

        This is the backstop for a state machine that failed to lock — the case the fail-safe
        path in PnLTracker.trip explicitly tolerates.
        """
        clock = _clock()
        breached = PnLTracker(
            StateMachine(TradingState.ACTIVE, clock=clock),
            daily_lock=DailyLock(path=tmp_path / "other_lock.txt", clock=clock),
            clock=clock,
        )
        breached.book_realised(Decimal("-600"))
        assert breached.is_breached

        # Remove the on-disk lock so check 3 passes and check 6 is reached: this proves the
        # in-memory latch alone refuses the trade, with no help from the state or the disk.
        breached.daily_lock.path.unlink()

        still_active = StateMachine(TradingState.ACTIVE, clock=clock)
        monitor = FeedMonitor(clock=clock)
        monitor.record()
        engine = RiskEngine(still_active, breached, monitor, PositionRegistry(), clock=clock)

        decision = engine.evaluate("RELIANCE")
        assert decision.reason is VetoReason.LOSS_LIMIT_BREACHED

    def test_vetoes_when_a_position_is_open(self, harness: _Harness) -> None:
        harness.positions.record_entry("INFY", quantity=10, at=TRADING_DAY)
        decision = harness.engine.evaluate("RELIANCE")
        assert decision.reason is VetoReason.POSITION_ALREADY_OPEN
        assert "INFY" in decision.detail

    def test_vetoes_symbols_outside_the_watchlist(self, harness: _Harness) -> None:
        decision = harness.engine.evaluate("NOT_A_REAL_SYMBOL")
        assert decision.reason is VetoReason.SYMBOL_NOT_ALLOWED

    def test_vetoes_during_reentry_cooldown(self, harness: _Harness) -> None:
        harness.positions.record_entry("RELIANCE", quantity=10, at=TRADING_DAY)
        harness.positions.record_exit("RELIANCE", was_stop_out=True, at=TRADING_DAY)
        decision = harness.engine.evaluate("RELIANCE")
        assert decision.reason is VetoReason.REENTRY_COOLDOWN

    def test_cooldown_expires_after_thirty_minutes(self, tmp_path: Path) -> None:
        harness = _Harness(tmp_path, _clock(11, 0))
        harness.positions.record_entry("RELIANCE", quantity=10, at=TRADING_DAY)
        harness.positions.record_exit(
            "RELIANCE", was_stop_out=True, at=TRADING_DAY - timedelta(minutes=31)
        )
        assert harness.engine.evaluate("RELIANCE").allowed

    def test_a_normal_exit_does_not_start_a_cooldown(self, harness: _Harness) -> None:
        """Only stop-outs cool down; taking profit must not block the next setup."""
        harness.positions.record_entry("RELIANCE", quantity=10, at=TRADING_DAY)
        harness.positions.record_exit("RELIANCE", was_stop_out=False, at=TRADING_DAY)
        assert harness.engine.evaluate("RELIANCE").allowed

    def test_vetoes_when_engine_is_cold(self, harness: _Harness) -> None:
        reset_warm_state()
        try:
            decision = harness.engine.evaluate("RELIANCE")
            assert decision.reason is VetoReason.ENGINE_COLD
        finally:
            assert warmup() is True

    def test_short_circuits_on_the_first_failure(self, harness: _Harness) -> None:
        """Cold engine is check 1; it must win over a later, also-failing check."""
        harness.machine.lock_out("test")  # would fail check 2
        reset_warm_state()
        try:
            assert harness.engine.evaluate("RELIANCE").reason is VetoReason.ENGINE_COLD
        finally:
            assert warmup() is True

    def test_records_which_check_failed(self, harness: _Harness) -> None:
        harness.clock.advance(5.0)
        decision = harness.engine.evaluate("RELIANCE")
        assert decision.failed_check == "feed_fresh"
        assert decision.detail

    def test_veto_counts_accumulate(self, harness: _Harness) -> None:
        harness.clock.advance(5.0)
        for _ in range(3):
            harness.engine.evaluate("RELIANCE")
        assert harness.engine.veto_counts[VetoReason.FEED_STALE] == 3


# ──────────────────────────────────────────────────────────────────────────────
# Fail-safe behaviour — the point of the module
# ──────────────────────────────────────────────────────────────────────────────


class TestFailsSafe:
    def test_a_check_that_raises_becomes_a_veto(self, harness: _Harness) -> None:
        """The core contract: a broken check must never mean permission."""

        class ExplodingMonitor:
            stale_after = 2.0

            @property
            def age(self) -> float:
                raise RuntimeError("monitor exploded")

        harness.engine = RiskEngine(
            harness.machine,
            harness.pnl,
            harness.monitor,
            harness.positions,
            clock=harness.clock,
        )
        harness.engine._feed_monitor = ExplodingMonitor()  # type: ignore[assignment]

        decision = harness.engine.evaluate("RELIANCE")
        assert not decision.allowed
        assert decision.reason is VetoReason.CHECK_FAILED
        assert "monitor exploded" in decision.detail

    @pytest.mark.parametrize(
        "check_method",
        [
            "_check_engine_warm",
            "_check_state_active",
            "_check_daily_lock",
            "_check_entry_window",
            "_check_feed_fresh",
            "_check_loss_limit",
            "_check_flat",
            "_check_symbol_allowed",
            "_check_cooldown",
            "_check_sentinel",
            "_check_margin",
        ],
    )
    def test_every_check_vetoes_when_it_raises(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch, check_method: str
    ) -> None:
        """Sweep all eleven: break each in turn, and none may let a trade through."""

        def exploder(*_args: object, **_kwargs: object) -> Any:
            raise RuntimeError(f"{check_method} exploded")

        monkeypatch.setattr(RiskEngine, check_method, exploder)

        decision = harness.engine.evaluate("RELIANCE")
        assert not decision.allowed, f"{check_method} raised but the gate allowed the trade"
        assert decision.reason is VetoReason.CHECK_FAILED
        assert check_method.removeprefix("_check_") in decision.detail or decision.failed_check

    def test_nan_realised_pnl_vetoes(self, harness: _Harness) -> None:
        """`Decimal('NaN') <= x` raises rather than returning False — must not slip through."""
        harness.pnl.book_realised(Decimal("NaN"))
        decision = harness.engine.evaluate("RELIANCE")
        assert not decision.allowed


class TestLossLimitSecondLineOfDefence:
    """The gate re-derives the breach rather than only trusting the tracker's latch.

    In practice ``PnLTracker`` trips first and the state check refuses earlier, so these
    branches never fire in a healthy system. That is exactly why they are worth proving: they
    are what stands between us and the market if the latch itself is ever bypassed.
    """

    def _with_snapshot(self, harness: _Harness, reading: Any) -> RiskEngine:
        class Stub:
            # NB: the attribute is deliberately not called `snapshot` in the closure — a class
            # body that also defines `snapshot()` shadows the enclosing name at class scope.
            is_breached = False
            daily_lock = harness.lock
            headroom = reading.headroom

            def snapshot(self) -> Any:
                return reading

        return RiskEngine(
            harness.machine,
            Stub(),  # type: ignore[arg-type]
            harness.monitor,
            harness.positions,
            clock=harness.clock,
        )

    def test_undefined_pnl_vetoes_even_with_the_latch_clear(self, harness: _Harness) -> None:
        from tachyon.risk.tracker import PnLSnapshot

        snapshot = PnLSnapshot(
            realised=Decimal("NaN"),
            floating=Decimal("0"),
            charges=Decimal("0"),
            total=Decimal("NaN"),
            headroom=Decimal("500"),
            breached=False,
            limit=DAILY_LOSS_LIMIT_INR,
        )
        decision = self._with_snapshot(harness, snapshot).evaluate("RELIANCE")
        assert not decision.allowed
        assert decision.reason is VetoReason.LOSS_LIMIT_BREACHED
        assert "undefined" in decision.detail

    def test_exhausted_headroom_vetoes_even_with_the_latch_clear(self, harness: _Harness) -> None:
        from tachyon.risk.tracker import PnLSnapshot

        snapshot = PnLSnapshot(
            realised=Decimal("-500"),
            floating=Decimal("0"),
            charges=Decimal("0"),
            total=Decimal("-500"),
            headroom=Decimal("0"),
            breached=False,
            limit=DAILY_LOSS_LIMIT_INR,
        )
        decision = self._with_snapshot(harness, snapshot).evaluate("RELIANCE")
        assert not decision.allowed
        assert decision.reason is VetoReason.LOSS_LIMIT_BREACHED
        assert "no headroom" in decision.detail

    def test_a_broken_clock_still_produces_a_decision(self, harness: _Harness) -> None:
        """The decision record must survive a clock that cannot be read."""

        class BrokenClock:
            def now(self) -> datetime:
                raise OSError("clock unavailable")

            def monotonic(self) -> float:
                return 0.0

        engine = RiskEngine(
            harness.machine,
            harness.pnl,
            harness.monitor,
            harness.positions,
            clock=BrokenClock(),  # type: ignore[arg-type]
        )
        decision = engine.evaluate("RELIANCE")
        assert not decision.allowed  # the entry-window check raises and vetoes
        assert decision.at_ist == datetime.min


class TestMarginCheck:
    """Check 10 — CLAUDE.md §4, added in Phase 7 alongside the execution layer."""

    def _engine(self, harness: _Harness, provider: Any) -> RiskEngine:
        return RiskEngine(
            harness.machine,
            harness.pnl,
            harness.monitor,
            harness.positions,
            clock=harness.clock,
            margin_provider=provider,
        )

    def test_passes_when_no_provider_is_wired(self, harness: _Harness) -> None:
        """An absent provider is an explicit configuration, not unknown state."""
        assert harness.engine.evaluate("RELIANCE").allowed

    def test_allows_with_positive_margin(self, harness: _Harness) -> None:
        engine = self._engine(harness, lambda: Decimal("25000"))
        assert engine.evaluate("RELIANCE").allowed

    @pytest.mark.parametrize("margin", [Decimal("0"), Decimal("-1"), Decimal("-99999")])
    def test_vetoes_on_non_positive_margin(self, harness: _Harness, margin: Decimal) -> None:
        engine = self._engine(harness, lambda: margin)
        decision = engine.evaluate("RELIANCE")
        assert not decision.allowed
        assert decision.reason is VetoReason.INSUFFICIENT_MARGIN

    @pytest.mark.parametrize("margin", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
    def test_vetoes_on_undefined_margin(self, harness: _Harness, margin: Decimal) -> None:
        """A NaN margin compares False against every threshold — it must be caught explicitly."""
        engine = self._engine(harness, lambda: margin)
        decision = engine.evaluate("RELIANCE")
        assert not decision.allowed
        assert decision.reason is VetoReason.INSUFFICIENT_MARGIN

    def test_vetoes_when_requirement_exceeds_available(self, harness: _Harness) -> None:
        engine = self._engine(harness, lambda: Decimal("5000"))
        decision = engine.evaluate("RELIANCE", required_margin=Decimal("30000"))
        assert not decision.allowed
        assert decision.reason is VetoReason.INSUFFICIENT_MARGIN
        assert "30000" in decision.detail

    def test_allows_when_requirement_is_covered(self, harness: _Harness) -> None:
        engine = self._engine(harness, lambda: Decimal("30000"))
        assert engine.evaluate("RELIANCE", required_margin=Decimal("29999.95")).allowed

    def test_provider_that_raises_becomes_a_veto(self, harness: _Harness) -> None:
        def broken() -> Decimal:
            raise ConnectionError("broker unreachable")

        decision = self._engine(harness, broken).evaluate("RELIANCE")
        assert not decision.allowed
        assert decision.reason is VetoReason.CHECK_FAILED

    def test_margin_is_evaluated_last(self, harness: _Harness) -> None:
        """A cheaper veto must short-circuit before anything touches the network."""
        calls: list[int] = []

        def counting() -> Decimal:
            calls.append(1)
            return Decimal("25000")

        engine = self._engine(harness, counting)
        harness.machine.lock_out("test")
        assert not engine.evaluate("RELIANCE").allowed
        assert calls == []

    def test_nan_floating_pnl_vetoes(self, harness: _Harness) -> None:
        harness.pnl.update_floating(Decimal("NaN"))
        assert not harness.engine.evaluate("RELIANCE").allowed

    def test_infinite_pnl_vetoes(self, harness: _Harness) -> None:
        harness.pnl.update_floating(Decimal("-Infinity"))
        assert not harness.engine.evaluate("RELIANCE").allowed

    def test_nan_pnl_trips_the_kill_switch(self, harness: _Harness) -> None:
        harness.pnl.book_realised(Decimal("NaN"))
        assert harness.pnl.is_breached
        assert harness.machine.state is TradingState.LOCKED

    def test_unreadable_lock_file_vetoes(self, harness: _Harness) -> None:
        """An unparseable lock counts as engaged (CLAUDE.md §1.2)."""
        harness.lock.path.write_text("\x00\x01 not a date", encoding="utf-8")
        decision = harness.engine.evaluate("RELIANCE")
        assert decision.reason is VetoReason.DAILY_LOCK_ENGAGED

    def test_evaluate_never_raises(self, harness: _Harness) -> None:
        """Even with every collaborator sabotaged, the gate returns a decision."""

        class Sabotage:
            def __getattr__(self, _name: str) -> Any:
                raise RuntimeError("everything is broken")

        harness.engine._state_machine = Sabotage()  # type: ignore[assignment]
        harness.engine._pnl = Sabotage()  # type: ignore[assignment]
        harness.engine._positions = Sabotage()  # type: ignore[assignment]
        harness.engine._feed_monitor = Sabotage()  # type: ignore[assignment]

        decision = harness.engine.evaluate("RELIANCE")
        assert isinstance(decision, RiskDecision)
        assert not decision.allowed

    def test_lock_file_write_failure_still_locks_this_session(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read-only disk must not stop the in-memory lock from engaging."""

        def deny(*_args: object, **_kwargs: object) -> None:
            raise PermissionError("read-only filesystem")

        monkeypatch.setattr(type(harness.lock), "engage", deny)

        harness.pnl.book_realised(Decimal("-600"))

        assert harness.pnl.is_breached, "the in-memory latch must engage regardless"
        assert harness.machine.state is TradingState.LOCKED
        assert not harness.pnl.lock_file_written, "and must report the durability gap"
        assert not harness.engine.can_open_position("RELIANCE")

    def test_trip_never_raises_even_if_state_machine_is_broken(self, harness: _Harness) -> None:
        class BrokenMachine:
            def is_terminal(self) -> bool:
                return False

            def lock_out(self, reason: str) -> None:
                raise RuntimeError("state machine broken")

        harness.pnl._state_machine = BrokenMachine()  # type: ignore[assignment]
        harness.pnl.trip("TEST", "forced")  # must not raise
        assert harness.pnl.is_breached


# ──────────────────────────────────────────────────────────────────────────────
# PnLTracker
# ──────────────────────────────────────────────────────────────────────────────


class TestPnLTracker:
    def test_starts_flat_with_full_headroom(self, harness: _Harness) -> None:
        assert harness.pnl.total == Decimal("0")
        assert harness.pnl.headroom == DAILY_LOSS_LIMIT_INR
        assert not harness.pnl.is_breached

    def test_realised_breach_trips_and_writes_the_lock(self, harness: _Harness) -> None:
        harness.pnl.book_realised(Decimal("-500"))

        assert harness.pnl.is_breached
        assert harness.machine.state is TradingState.LOCKED
        assert harness.lock.path.is_file()
        assert harness.lock.path.read_text(encoding="utf-8").splitlines()[0] == "2026-08-10"
        assert harness.pnl.lock_file_written

    def test_floating_breach_trips_too(self, harness: _Harness) -> None:
        """An open loser must not be allowed to run past the limit unbooked."""
        harness.pnl.update_floating(Decimal("-501"))
        assert harness.pnl.is_breached
        assert harness.machine.state is TradingState.LOCKED

    def test_exactly_at_the_limit_trips(self, harness: _Harness) -> None:
        harness.pnl.book_realised(-DAILY_LOSS_LIMIT_INR)
        assert harness.pnl.is_breached

    def test_one_rupee_short_does_not_trip(self, harness: _Harness) -> None:
        harness.pnl.book_realised(Decimal("-499"))
        assert not harness.pnl.is_breached
        assert harness.pnl.headroom == Decimal("1")

    def test_charges_count_toward_the_limit(self, harness: _Harness) -> None:
        harness.pnl.book_realised(Decimal("-480"))
        assert not harness.pnl.is_breached
        harness.pnl.add_charges(Decimal("25"))
        assert harness.pnl.is_breached, "charges are real money (CLAUDE.md §1.2)"

    def test_the_latch_does_not_release(self, harness: _Harness) -> None:
        harness.pnl.update_floating(Decimal("-600"))
        assert harness.pnl.is_breached

        harness.pnl.update_floating(Decimal("+1000"))
        assert harness.pnl.is_breached, "a latching switch never un-trips"
        assert harness.machine.state is TradingState.LOCKED

    def test_trip_is_idempotent(self, harness: _Harness) -> None:
        harness.pnl.book_realised(Decimal("-600"))
        harness.pnl.book_realised(Decimal("-600"))
        harness.pnl.trip("AGAIN")
        assert harness.machine.state is TradingState.LOCKED

    def test_headroom_never_goes_negative(self, harness: _Harness) -> None:
        harness.pnl.book_realised(Decimal("-9999"))
        assert harness.pnl.headroom == Decimal("0")

    def test_money_is_decimal_not_float(self, harness: _Harness) -> None:
        harness.pnl.book_realised(Decimal("-0.1"))
        harness.pnl.book_realised(Decimal("-0.2"))
        assert harness.pnl.realised == Decimal("-0.3"), "float would give -0.30000000000000004"

    def test_lock_file_records_the_mode(self, tmp_path: Path) -> None:
        clock = _clock()
        machine = StateMachine(TradingState.ACTIVE, clock=clock)
        lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)
        pnl = PnLTracker(machine, daily_lock=lock, mode=TradingMode.PAPER, clock=clock)
        pnl.book_realised(Decimal("-500"))

        record = lock.read()
        assert record is not None
        assert record.mode is TradingMode.PAPER

    def test_concurrent_booking_is_consistent(self, harness: _Harness) -> None:
        def book() -> None:
            for _ in range(200):
                harness.pnl.book_realised(Decimal("-1"))

        threads = [threading.Thread(target=book) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert harness.pnl.realised == Decimal("-800")


class TestPositionRegistry:
    def test_tracks_open_positions(self) -> None:
        registry = PositionRegistry()
        assert registry.open_count == 0

        registry.record_entry("RELIANCE", quantity=10, at=TRADING_DAY)
        assert registry.open_count == 1
        assert registry.is_open("RELIANCE")

        registry.record_exit("RELIANCE", was_stop_out=False, at=TRADING_DAY)
        assert registry.open_count == 0

    def test_cooldown_window(self) -> None:
        registry = PositionRegistry()
        registry.record_exit("RELIANCE", was_stop_out=True, at=TRADING_DAY)

        assert registry.in_cooldown("RELIANCE", TRADING_DAY + timedelta(minutes=29))
        assert not registry.in_cooldown("RELIANCE", TRADING_DAY + timedelta(minutes=30))
        assert registry.cooldown_until("RELIANCE") == TRADING_DAY + timedelta(minutes=30)

    def test_cooldown_is_per_symbol(self) -> None:
        registry = PositionRegistry()
        registry.record_exit("RELIANCE", was_stop_out=True, at=TRADING_DAY)
        assert not registry.in_cooldown("INFY", TRADING_DAY)


# ──────────────────────────────────────────────────────────────────────────────
# SquareOffWatchdog
# ──────────────────────────────────────────────────────────────────────────────


class TestSquareOffWatchdog:
    def test_fires_at_the_deadline(self) -> None:
        clock = _clock(15, 14, 30)
        machine = StateMachine(TradingState.ACTIVE, clock=clock)
        fired: list[str] = []

        watchdog = SquareOffWatchdog(
            machine,
            on_square_off=lambda: fired.append("flat"),
            clock=clock,
            tick_seconds=0.01,
        )
        watchdog.start()
        try:
            clock.advance(31)
            _wait_until(lambda: watchdog.square_off_completed)
        finally:
            watchdog.stop()

        assert watchdog.square_off_fired
        assert fired == ["flat"]
        assert machine.state is TradingState.SQUARING_OFF

    def test_does_not_fire_early(self) -> None:
        clock = _clock(15, 14, 0)
        machine = StateMachine(TradingState.ACTIVE, clock=clock)
        watchdog = SquareOffWatchdog(machine, clock=clock, tick_seconds=0.01)
        watchdog.start()
        try:
            clock.advance(30)
            _sleep_ticks()
            assert not watchdog.square_off_fired
        finally:
            watchdog.stop()

    def test_wall_clock_jump_backward_cannot_postpone_it(self) -> None:
        """The deadline is pinned to monotonic time at arm; NTP cannot move it."""
        clock = _clock(15, 14, 30)
        machine = StateMachine(TradingState.ACTIVE, clock=clock)
        watchdog = SquareOffWatchdog(machine, clock=clock, tick_seconds=0.01)
        watchdog.start()
        try:
            clock.jump_wall_clock(-3600)  # wall clock now reads 14:14
            clock.advance(31)  # but 31 real seconds still pass
            _wait_until(lambda: watchdog.square_off_fired)
        finally:
            watchdog.stop()

        assert watchdog.square_off_fired

    def test_retries_a_failing_square_off_until_it_succeeds(self) -> None:
        """A broker timeout at 15:15 must not leave a position open overnight."""
        clock = _clock(15, 14, 30)
        machine = StateMachine(TradingState.ACTIVE, clock=clock)
        attempts = {"n": 0}

        def flaky() -> None:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("broker timeout")

        watchdog = SquareOffWatchdog(machine, on_square_off=flaky, clock=clock, tick_seconds=0.01)
        watchdog.start()
        try:
            clock.advance(31)
            _wait_until(lambda: watchdog.square_off_completed, advance=clock, step=3.0)
        finally:
            watchdog.stop()

        assert watchdog.square_off_completed
        assert attempts["n"] == 3
        assert watchdog.action_failures == 2

    def test_thread_survives_a_failing_action(self) -> None:
        clock = _clock(15, 14, 30)
        machine = StateMachine(TradingState.ACTIVE, clock=clock)

        def always_fails() -> None:
            raise RuntimeError("permanently broken")

        watchdog = SquareOffWatchdog(
            machine, on_square_off=always_fails, clock=clock, tick_seconds=0.01
        )
        watchdog.start()
        try:
            clock.advance(31)
            _wait_until(lambda: watchdog.action_failures >= 2, advance=clock, step=3.0)
            assert watchdog.is_alive, "the watchdog must be immortal"
        finally:
            watchdog.stop()

    def test_fires_even_when_the_state_machine_is_locked(self) -> None:
        """A kill switch earlier in the day must not cancel the flatten."""
        clock = _clock(15, 14, 30)
        machine = StateMachine(TradingState.ACTIVE, clock=clock)
        machine.lock_out("earlier breach")
        fired: list[str] = []

        watchdog = SquareOffWatchdog(
            machine, on_square_off=lambda: fired.append("flat"), clock=clock, tick_seconds=0.01
        )
        watchdog.start()
        try:
            clock.advance(31)
            _wait_until(lambda: watchdog.square_off_completed)
        finally:
            watchdog.stop()

        assert fired == ["flat"], "flattening must happen regardless of the state label"
        assert machine.state is TradingState.LOCKED

    def test_fires_immediately_when_started_after_the_deadline(self) -> None:
        """Booting at 15:20 must flatten now, not wait until tomorrow."""
        clock = _clock(15, 20, 0)
        machine = StateMachine(TradingState.ACTIVE, clock=clock)
        fired: list[str] = []

        watchdog = SquareOffWatchdog(
            machine, on_square_off=lambda: fired.append("flat"), clock=clock, tick_seconds=0.01
        )
        watchdog.start()
        try:
            _wait_until(lambda: watchdog.square_off_completed)
        finally:
            watchdog.stop()

        assert fired == ["flat"]

    def test_drives_the_no_new_entries_transition(self) -> None:
        clock = _clock(14, 59, 30)
        machine = StateMachine(TradingState.ACTIVE, clock=clock)
        watchdog = SquareOffWatchdog(machine, clock=clock, tick_seconds=0.01)
        watchdog.start()
        try:
            clock.advance(31)
            _wait_until(lambda: machine.state is TradingState.NO_NEW_ENTRIES)
        finally:
            watchdog.stop()

        assert machine.state is TradingState.NO_NEW_ENTRIES
        assert watchdog.has_fired(SessionEvent.NO_NEW_ENTRIES)

    def test_start_is_idempotent(self) -> None:
        clock = _clock(11, 0)
        watchdog = SquareOffWatchdog(
            StateMachine(TradingState.ACTIVE, clock=clock), clock=clock, tick_seconds=0.01
        )
        watchdog.start()
        watchdog.start()
        try:
            assert watchdog.is_alive
        finally:
            watchdog.stop()
        assert not watchdog.is_alive

    def test_stop_is_safe_before_start(self) -> None:
        clock = _clock(11, 0)
        watchdog = SquareOffWatchdog(StateMachine(TradingState.ACTIVE, clock=clock), clock=clock)
        watchdog.stop()  # must not raise

    def test_no_unexpected_thread_errors(self) -> None:
        clock = _clock(15, 14, 30)
        machine = StateMachine(TradingState.ACTIVE, clock=clock)
        watchdog = SquareOffWatchdog(
            machine, on_square_off=lambda: None, clock=clock, tick_seconds=0.01
        )
        watchdog.start()
        try:
            clock.advance(31)
            _wait_until(lambda: watchdog.square_off_completed)
        finally:
            watchdog.stop()
        assert watchdog.thread_errors == 0


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def _wait_until(
    predicate: object,
    limit_seconds: float = 5.0,
    advance: ManualClock | None = None,
    step: float = 0.0,
) -> None:
    """Spin until ``predicate`` holds, optionally advancing a ManualClock as we go."""
    import time as _time

    deadline = _time.monotonic() + limit_seconds
    while _time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return
        if advance is not None and step:
            advance.advance(step)
        _time.sleep(0.005)
    raise AssertionError("condition not reached within the time limit")


def _sleep_ticks(seconds: float = 0.15) -> None:
    import time as _time

    _time.sleep(seconds)
