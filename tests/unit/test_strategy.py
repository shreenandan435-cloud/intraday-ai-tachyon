"""Phase 8 strategy tests — CLAUDE.md §3.1, §4, §8.1.

Three things are worth proving here, and they are not the happy path.

**The confluence rule is exhaustive.** Seven of the eight combinations of (VWAP, EMA, OBI) must
produce NEUTRAL. A rule that fires on two of three is a different, much looser strategy, and the
difference is invisible in a live session until the losses arrive.

**Refusals actually refuse.** The Brain is a chain of gates: cooldown, risk, sizing. Each one is
broken in turn and asserted to stop the order — not merely to log about it.

**The loop does not block.** A broker that takes a second to answer must not stop tick
ingestion, because a stalled consumer produces a sequence gap, and a sequence gap voids the
session VWAP for the rest of the day (CLAUDE.md §3.3).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from tachyon.core.clock import IST, ManualClock, now_ist
from tachyon.core.config import (
    CapitalSettings,
    ExecutionSettings,
    MathEngineSettings,
    Settings,
    StrategySettings,
    WatchlistItem,
)
from tachyon.core.constants import REENTRY_COOLDOWN, TradingMode
from tachyon.core.state import DailyLock, StateMachine, TradingState
from tachyon.execution.builder import OrderBuilder, Side
from tachyon.execution.executor import (
    ExecutionReport,
    LegResult,
    NotFlatError,
    RoboExecutor,
)
from tachyon.execution.journal import OrderJournal
from tachyon.ipc.monitor import FeedMonitor
from tachyon.ipc.schemas import Heartbeat, OrderBook, Tick, depth_topic, tick_topic
from tachyon.ipc.subscriber import Envelope
from tachyon.math_engine import warmup
from tachyon.math_engine.core import IndicatorSnapshot, TickAggregator
from tachyon.risk.engine import RiskEngine, VetoReason
from tachyon.risk.tracker import PnLTracker, PositionRegistry
from tachyon.strategy.brain import StrategyBrain
from tachyon.strategy.cooldown import (
    REENTRY_COOLDOWN_MINUTES,
    CooldownTooShortError,
    ReentryManager,
)
from tachyon.strategy.signals import NoSignalReason, Signal, SignalGenerator
from tachyon.strategy.telemetry import StatePublisher
from tachyon.ui.postback import OrderUpdate
from tachyon.utils.telegram_alerts import TelegramAlerter


@pytest.fixture(scope="session", autouse=True)
def _warm_engine() -> None:
    assert warmup() is True


def _clock(hh: int = 11, mm: int = 0, ss: int = 0, mono: float = 1000.0) -> ManualClock:
    return ManualClock(wall=datetime(2026, 8, 10, hh, mm, ss, tzinfo=IST), mono=mono)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "watchlist": (
            WatchlistItem(symbol="RELIANCE", token="2885", exchange="NSE"),
            WatchlistItem(symbol="HDFCBANK", token="1333", exchange="NSE"),
        ),
        "math_engine": MathEngineSettings(),
        "strategy": StrategySettings(),
        "execution": ExecutionSettings(),
    }
    base.update(overrides)
    return Settings(**base)


def _snapshot(
    *,
    symbol: str = "RELIANCE",
    ltp: float = 2510.0,
    session_vwap: float = 2500.0,
    ema: float = 2505.0,
    ema_period: int = 21,
    obi: float = 0.5,
    obi_weighted: float = 0.5,
    atr_5m: float = 8.4,
    tainted: bool = False,
    session_vwap_valid: bool = True,
) -> IndicatorSnapshot:
    """A snapshot that is LONG by default. Every test perturbs exactly one field."""
    return IndicatorSnapshot(
        symbol=symbol,
        ts_epoch=1_786_000_000.0,
        ltp=ltp,
        session_vwap=session_vwap,
        rolling_vwap=session_vwap,
        ema_periods=(9, ema_period, 50),
        emas=(ema, ema, ema),
        atr_5m=atr_5m,
        obi=obi,
        obi_weighted=obi_weighted,
        tick_count=500,
        candle_count=30,
        tainted=tainted,
        session_vwap_valid=session_vwap_valid,
        gaps_detected=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# signals.py — the confluence rule
# ──────────────────────────────────────────────────────────────────────────────


class TestSignalGenerator:
    @pytest.fixture
    def generator(self) -> SignalGenerator:
        return SignalGenerator(settings=_settings())

    def test_long_on_full_confluence(self, generator: SignalGenerator) -> None:
        report = generator.evaluate(
            _snapshot(ltp=2510.0, session_vwap=2500.0, ema=2505.0, obi=0.5, obi_weighted=0.45)
        )
        assert report.signal is Signal.LONG
        assert report.reason is NoSignalReason.NONE
        assert bool(report)

    def test_short_on_full_confluence(self, generator: SignalGenerator) -> None:
        report = generator.evaluate(
            _snapshot(ltp=2490.0, session_vwap=2500.0, ema=2495.0, obi=-0.5, obi_weighted=-0.45)
        )
        assert report.signal is Signal.SHORT
        assert report.reason is NoSignalReason.NONE

    @pytest.mark.parametrize(
        ("above_vwap", "above_ema", "obi_positive"),
        [
            (True, True, False),
            (True, False, True),
            (True, False, False),
            (False, True, True),
            (False, True, False),
            (False, False, True),
            (False, False, False),
        ],
    )
    def test_every_partial_confluence_is_neutral(
        self,
        generator: SignalGenerator,
        above_vwap: bool,
        above_ema: bool,
        obi_positive: bool,
    ) -> None:
        """Only 3-of-3 fires. Any two agreeing is a chop, not an edge."""
        ltp = 2510.0
        report = generator.evaluate(
            _snapshot(
                ltp=ltp,
                session_vwap=2500.0 if above_vwap else 2520.0,
                ema=2505.0 if above_ema else 2515.0,
                obi=0.5 if obi_positive else 0.0,
                obi_weighted=0.5 if obi_positive else 0.0,
            )
        )
        assert report.signal is Signal.NEUTRAL

    @pytest.mark.parametrize(
        ("obi", "expected"),
        [
            (0.31, Signal.LONG),
            (0.30, Signal.NEUTRAL),  # strictly greater than the threshold
            (0.29, Signal.NEUTRAL),
            (0.0, Signal.NEUTRAL),
            (1.0, Signal.LONG),
        ],
    )
    def test_obi_threshold_boundary(
        self, generator: SignalGenerator, obi: float, expected: Signal
    ) -> None:
        report = generator.evaluate(_snapshot(obi=obi, obi_weighted=obi))
        assert report.signal is expected

    @pytest.mark.parametrize(
        ("obi", "expected"),
        [(-0.31, Signal.SHORT), (-0.30, Signal.NEUTRAL), (-0.29, Signal.NEUTRAL)],
    )
    def test_obi_threshold_boundary_short(
        self, generator: SignalGenerator, obi: float, expected: Signal
    ) -> None:
        report = generator.evaluate(
            _snapshot(ltp=2490.0, session_vwap=2500.0, ema=2495.0, obi=obi, obi_weighted=obi)
        )
        assert report.signal is expected

    def test_threshold_is_a_tunable(self) -> None:
        loose = SignalGenerator(settings=_settings(), obi_threshold=0.1)
        assert loose.evaluate(_snapshot(obi=0.2, obi_weighted=0.2)).signal is Signal.LONG

        tight = SignalGenerator(settings=_settings(), obi_threshold=0.8)
        assert tight.evaluate(_snapshot(obi=0.5, obi_weighted=0.5)).signal is Signal.NEUTRAL

    def test_default_threshold_is_point_three(self, generator: SignalGenerator) -> None:
        assert generator.obi_threshold == 0.3

    @pytest.mark.parametrize("threshold", [0.0, 1.0, 1.5, -0.3])
    def test_rejects_a_meaningless_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="must lie in"):
            SignalGenerator(settings=_settings(), obi_threshold=threshold)

    # ── the anti-spoofing check ──────────────────────────────────────────────

    def test_obi_disagreement_suppresses_the_entry(self, generator: SignalGenerator) -> None:
        """Bid-heavy at the touch, ask-heavy in the ladder is the classic spoof shape."""
        report = generator.evaluate(_snapshot(obi=0.6, obi_weighted=0.05))
        assert report.signal is Signal.NEUTRAL
        assert report.reason is NoSignalReason.OBI_DISAGREEMENT

    def test_agreement_can_be_disabled(self) -> None:
        permissive = SignalGenerator(settings=_settings(), require_obi_agreement=False)
        assert permissive.evaluate(_snapshot(obi=0.6, obi_weighted=0.05)).signal is Signal.LONG

    # ── NaN handling ─────────────────────────────────────────────────────────

    def test_nan_session_vwap_blocks_every_signal(self, generator: SignalGenerator) -> None:
        """After a sequence gap this is permanent for the session (CLAUDE.md §3.3)."""
        report = generator.evaluate(_snapshot(session_vwap=math.nan, session_vwap_valid=False))
        assert report.signal is Signal.NEUTRAL
        assert report.reason is NoSignalReason.SESSION_VWAP_UNAVAILABLE

    def test_nan_ema_blocks_every_signal(self, generator: SignalGenerator) -> None:
        report = generator.evaluate(_snapshot(ema=math.nan))
        assert report.signal is Signal.NEUTRAL
        assert report.reason is NoSignalReason.EMA_UNAVAILABLE

    @pytest.mark.parametrize("atr", [math.nan, 0.0, -1.0, math.inf])
    def test_unusable_atr_blocks_every_signal(self, generator: SignalGenerator, atr: float) -> None:
        """No ATR means no stop, no target and no size — so no legitimate order."""
        report = generator.evaluate(_snapshot(atr_5m=atr))
        assert report.signal is Signal.NEUTRAL
        assert report.reason is NoSignalReason.ENGINE_NOT_READY

    def test_tainted_buffers_block_every_signal(self, generator: SignalGenerator) -> None:
        report = generator.evaluate(_snapshot(tainted=True))
        assert report.signal is Signal.NEUTRAL
        assert report.reason is NoSignalReason.ENGINE_NOT_READY

    def test_nan_ltp_blocks_every_signal(self, generator: SignalGenerator) -> None:
        assert generator.evaluate(_snapshot(ltp=math.nan)).signal is Signal.NEUTRAL

    def test_price_exactly_on_vwap_is_neutral(self, generator: SignalGenerator) -> None:
        report = generator.evaluate(_snapshot(ltp=2500.0, session_vwap=2500.0, ema=2490.0))
        assert report.signal is Signal.NEUTRAL
        assert report.reason is NoSignalReason.NO_VWAP_CONFLUENCE

    # ── EMA selection ────────────────────────────────────────────────────────

    def test_uses_the_configured_ema_period(self) -> None:
        snapshot = IndicatorSnapshot(
            symbol="RELIANCE",
            ts_epoch=0.0,
            ltp=2510.0,
            session_vwap=2500.0,
            rolling_vwap=2500.0,
            ema_periods=(9, 21, 50),
            emas=(2505.0, 2520.0, 2400.0),  # only the 21 sits above price
            atr_5m=8.4,
            obi=0.5,
            obi_weighted=0.5,
            tick_count=500,
            candle_count=30,
            tainted=False,
            session_vwap_valid=True,
            gaps_detected=0,
        )
        fast = SignalGenerator(settings=_settings(), ema_period=9)
        assert fast.evaluate(snapshot).signal is Signal.LONG

        mid = SignalGenerator(settings=_settings(), ema_period=21)
        assert mid.evaluate(snapshot).signal is Signal.NEUTRAL

    def test_an_uncomputed_ema_period_degrades_to_neutral(self) -> None:
        """Config validation catches this at boot; the runtime backstop must not crash."""
        generator = SignalGenerator(settings=_settings(), ema_period=200)
        report = generator.evaluate(_snapshot())
        assert report.signal is Signal.NEUTRAL
        assert report.reason is NoSignalReason.EMA_UNAVAILABLE

    def test_counters_and_reset(self, generator: SignalGenerator) -> None:
        generator.evaluate(_snapshot())
        generator.evaluate(_snapshot(obi=0.0, obi_weighted=0.0))
        assert generator.counts[Signal.LONG] == 1
        assert generator.counts[Signal.NEUTRAL] == 1
        generator.reset_session()
        assert generator.counts[Signal.LONG] == 0

    def test_report_carries_its_inputs(self, generator: SignalGenerator) -> None:
        report = generator.evaluate(_snapshot())
        assert "RELIANCE" in report.detail
        assert "2510.00" in report.detail
        assert report.atr_5m == 8.4


class TestConfigValidation:
    def test_ema_period_must_be_computed(self) -> None:
        """A filter evaluated against an EMA nothing computes is silently no filter at all."""
        with pytest.raises(ValueError, match="not in"):
            _settings(
                math_engine=MathEngineSettings(ema_periods=(9, 50)),
                strategy=StrategySettings(ema_period=21),
            )

    def test_matching_periods_are_accepted(self) -> None:
        settings = _settings(
            math_engine=MathEngineSettings(ema_periods=(9, 21, 50)),
            strategy=StrategySettings(ema_period=50),
        )
        assert settings.strategy.ema_period == 50


# ──────────────────────────────────────────────────────────────────────────────
# cooldown.py
# ──────────────────────────────────────────────────────────────────────────────


class TestReentryManager:
    def test_constant_is_thirty_minutes(self) -> None:
        assert REENTRY_COOLDOWN_MINUTES == 30.0
        assert REENTRY_COOLDOWN == timedelta(minutes=30)

    def test_a_symbol_never_traded_is_free(self) -> None:
        manager = ReentryManager(clock=_clock())
        assert manager.may_enter("RELIANCE")
        assert manager.last_exit("RELIANCE") is None

    @pytest.mark.parametrize(
        ("elapsed_minutes", "allowed"),
        [(0, False), (1, False), (15, False), (29, False), (29.99, False), (30, True), (45, True)],
    )
    def test_the_window(self, elapsed_minutes: float, allowed: bool) -> None:
        clock = _clock()
        manager = ReentryManager(clock=clock)
        manager.record_exit("RELIANCE", was_stop_out=True)
        clock.advance(elapsed_minutes * 60.0)
        assert manager.may_enter("RELIANCE") is allowed

    def test_the_boundary_is_inclusive(self) -> None:
        """At exactly last_exit + 30:00 the window has elapsed, not "almost"."""
        clock = _clock()
        manager = ReentryManager(clock=clock)
        manager.record_exit("RELIANCE")
        clock.advance(REENTRY_COOLDOWN.total_seconds())
        assert manager.check("RELIANCE").remaining_seconds == 0.0
        assert manager.may_enter("RELIANCE")

    def test_cooldown_is_per_symbol(self) -> None:
        manager = ReentryManager(clock=_clock())
        manager.record_exit("RELIANCE", was_stop_out=True)
        assert not manager.may_enter("RELIANCE")
        assert manager.may_enter("HDFCBANK")

    def test_applies_to_winners_too_by_default(self) -> None:
        """Thrashing across VWAP does not care whether the last trade won."""
        manager = ReentryManager(clock=_clock())
        manager.record_exit("RELIANCE", was_stop_out=False)
        assert not manager.may_enter("RELIANCE")

    def test_stop_out_only_mode(self) -> None:
        manager = ReentryManager(clock=_clock(), apply_to_every_exit=False)
        manager.record_exit("RELIANCE", was_stop_out=False)
        assert manager.may_enter("RELIANCE")
        manager.record_exit("RELIANCE", was_stop_out=True)
        assert not manager.may_enter("RELIANCE")

    def test_cannot_be_shortened_below_the_constitutional_minimum(self) -> None:
        with pytest.raises(CooldownTooShortError):
            ReentryManager(timedelta(minutes=5))
        with pytest.raises(CooldownTooShortError):
            ReentryManager(timedelta(minutes=29, seconds=59))

    def test_may_be_lengthened(self) -> None:
        clock = _clock()
        manager = ReentryManager(timedelta(minutes=60), clock=clock)
        manager.record_exit("RELIANCE")
        clock.advance(31 * 60)
        assert not manager.may_enter("RELIANCE")
        clock.advance(30 * 60)
        assert manager.may_enter("RELIANCE")

    def test_verdict_reports_when_the_symbol_frees_up(self) -> None:
        clock = _clock(hh=11, mm=0)
        manager = ReentryManager(clock=clock)
        manager.record_exit("RELIANCE", was_stop_out=True)
        clock.advance(600)
        verdict = manager.check("RELIANCE")
        assert not verdict
        assert verdict.remaining_seconds == pytest.approx(1200.0)
        assert verdict.blocked_until_ist is not None
        assert verdict.blocked_until_ist.hour == 11
        assert verdict.blocked_until_ist.minute == 30
        assert "stop-out" in verdict.detail

    def test_cooling_symbols_and_counters(self) -> None:
        clock = _clock()
        manager = ReentryManager(clock=clock)
        manager.record_exit("RELIANCE")
        assert manager.cooling_symbols() == frozenset({"RELIANCE"})
        manager.may_enter("RELIANCE")
        assert manager.blocked_count == 1
        clock.advance(31 * 60)
        assert manager.cooling_symbols() == frozenset()

    def test_reset_session_clears_everything(self) -> None:
        manager = ReentryManager(clock=_clock())
        manager.record_exit("RELIANCE", was_stop_out=True)
        manager.reset_session()
        assert manager.may_enter("RELIANCE")
        assert manager.blocked_count == 0

    def test_there_is_no_per_symbol_escape_hatch(self) -> None:
        """The rule is most valuable exactly when an operator wants to override it."""
        assert not hasattr(ReentryManager, "clear")
        assert not hasattr(ReentryManager, "release")


# ──────────────────────────────────────────────────────────────────────────────
# brain.py
# ──────────────────────────────────────────────────────────────────────────────


class _FakeExecutor:
    """Records placements instead of making them, with a controllable delay."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[Any] = []
        self.delay = delay
        self.concurrent = 0
        self.max_concurrent = 0

    async def place_robo_order(self, plan: Any, decision: Any) -> ExecutionReport:
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.calls.append((plan, decision))
            return ExecutionReport(
                symbol=plan.symbol,
                side=plan.side,
                at_ist="now",
                plan=plan,
                results=tuple(
                    LegResult(
                        leg=leg.leg,
                        order_tag=leg.order_tag,
                        quantity=leg.quantity,
                        accepted=True,
                        order_id=f"FAKE-{leg.order_tag}",
                    )
                    for leg in plan.legs
                ),
                simulated=True,
            )
        finally:
            self.concurrent -= 1

    def square_off_action(self, loop: Any, timeout: float = 20.0) -> Any:
        return lambda: None


class _RecordingPublisher:
    """Stands in for a bound ZeroMQ PUB socket.

    Injected into every harness so that *constructing a Brain in a test binds no port*. Two
    brains in one process is normal here, and a real publisher would make the second one fight
    the first for tcp://127.0.0.1:5556 — and leave a context that never terminates.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes]] = []
        self.closed = False

    def publish_raw(self, topic: bytes, payload: bytes) -> None:
        self.sent.append((topic.decode(), payload))

    def close(self) -> None:
        self.closed = True

    def topics(self) -> list[str]:
        return [topic for topic, _payload in self.sent]


class _Harness:
    """A Brain wired to a fake executor, with helpers to drive it deterministically."""

    def __init__(self, tmp_path: Path, clock: ManualClock, **overrides: Any) -> None:
        from tachyon.strategy.brain import _SymbolState

        self.clock = clock
        self.settings = _settings(**overrides)
        self.machine = StateMachine(TradingState.ACTIVE, clock=clock)
        self.publisher = _RecordingPublisher()
        self.brain = StrategyBrain(
            settings=self.settings,
            client=None,
            subscriber=None,
            state_machine=self.machine,
            state_publisher=StatePublisher(
                settings=self.settings,
                publisher=self.publisher,  # type: ignore[arg-type]
                clock=clock,
            ),
            daily_lock=DailyLock(path=tmp_path / "daily_lock.txt", clock=clock),
            clock=clock,
        )
        self.brain._journal = OrderJournal(tmp_path / "journal", clock=clock)  # noqa: SLF001
        self.brain._monitor.record()  # noqa: SLF001

        for item in self.settings.watchlist:
            self.brain._symbols[item.symbol] = _SymbolState(  # noqa: SLF001
                aggregator=TickAggregator.from_settings(item.symbol, self.settings)
            )

        self.executor = _FakeExecutor()
        self.brain._executor = self.executor  # type: ignore[assignment]  # noqa: SLF001

    @property
    def placed(self) -> int:
        return len(self.executor.calls)

    def force_long(self, symbol: str = "RELIANCE") -> None:
        """Replace the generator so the next evaluation is unambiguously LONG."""

        class AlwaysLong:
            def evaluate(self, snapshot: IndicatorSnapshot) -> Any:
                from tachyon.strategy.signals import SignalReport

                return SignalReport(
                    symbol=symbol,
                    signal=Signal.LONG,
                    reason=NoSignalReason.NONE,
                    ltp=2500.0,
                    session_vwap=2490.0,
                    ema=2495.0,
                    ema_period=21,
                    obi=0.5,
                    obi_weighted=0.5,
                    obi_threshold=0.3,
                    atr_5m=8.4,
                    ts_epoch=0.0,
                )

        self.brain._generator = AlwaysLong()  # type: ignore[assignment]  # noqa: SLF001

    def evaluate(self, symbol: str = "RELIANCE") -> None:
        state = self.brain._symbols[symbol]  # noqa: SLF001
        self.brain._evaluate(symbol, state)  # noqa: SLF001

    async def settle(self) -> None:
        """Let dispatched entry tasks finish."""
        tasks = tuple(self.brain._entry_tasks)  # noqa: SLF001
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.fixture
def harness(tmp_path: Path) -> _Harness:
    return _Harness(tmp_path, _clock())


class TestBrainDecisionChain:
    async def test_places_an_order_on_a_clean_signal(self, harness: _Harness) -> None:
        harness.force_long()
        harness.evaluate()
        await harness.settle()
        assert harness.placed == 1
        assert harness.brain.stats.entries_placed == 1

    async def test_neutral_signal_places_nothing(self, harness: _Harness) -> None:
        harness.evaluate()  # real generator, empty aggregator -> ENGINE_NOT_READY
        await harness.settle()
        assert harness.placed == 0
        assert harness.brain.stats.signals == 0

    async def test_cooldown_drops_the_signal(self, harness: _Harness) -> None:
        harness.force_long()
        harness.brain.cooldown.record_exit("RELIANCE", was_stop_out=True)
        harness.evaluate()
        await harness.settle()
        assert harness.placed == 0
        assert harness.brain.stats.cooldown_blocks == 1
        assert harness.brain.stats.risk_vetoes == 0, "cooldown must short-circuit before risk"

    async def test_cooldown_expiry_lets_the_signal_through(self, harness: _Harness) -> None:
        harness.force_long()
        harness.brain.cooldown.record_exit("RELIANCE", was_stop_out=True)
        harness.evaluate()
        assert harness.placed == 0

        harness.clock.advance(REENTRY_COOLDOWN.total_seconds() + 1)
        # The monitor's staleness runs on the same monotonic clock, so a 30-minute jump also
        # makes the feed look dead. Re-arm it: this test is about the cooldown, not the feed.
        harness.brain.feed_monitor.record()
        harness.evaluate()
        await harness.settle()
        assert harness.placed == 1

    @pytest.mark.parametrize(
        ("break_it", "expected_reason"),
        [
            ("state", VetoReason.STATE_NOT_ACTIVE),
            ("feed", VetoReason.FEED_STALE),
            ("position", VetoReason.POSITION_ALREADY_OPEN),
            # Booking the limit also locks the state machine, and the state check runs earlier
            # (Phase 6). Two independent mechanisms refuse; the earlier one is what is reported.
            ("pnl", VetoReason.STATE_NOT_ACTIVE),
        ],
    )
    async def test_risk_veto_drops_the_signal(
        self, harness: _Harness, break_it: str, expected_reason: VetoReason
    ) -> None:
        harness.force_long()
        if break_it == "state":
            harness.machine.transition_to(TradingState.NO_NEW_ENTRIES, reason="test")
        elif break_it == "feed":
            harness.clock.advance(5.0)
        elif break_it == "position":
            harness.brain.positions.record_entry("HDFCBANK", 10)
        elif break_it == "pnl":
            harness.brain.pnl.book_realised(Decimal("-500"))

        harness.evaluate()
        await harness.settle()
        assert harness.placed == 0
        assert harness.brain.stats.risk_vetoes == 1
        assert harness.brain.risk.veto_counts.get(expected_reason) == 1

    async def test_outside_the_entry_window_drops_the_signal(self, tmp_path: Path) -> None:
        late = _Harness(tmp_path, _clock(hh=15, mm=5))
        late.force_long()
        late.evaluate()
        await late.settle()
        assert late.placed == 0
        assert late.brain.stats.risk_vetoes == 1

    async def test_unsizeable_order_is_dropped_not_rounded_up(self, harness: _Harness) -> None:
        """A budget that buys less than one share is no trade (CLAUDE.md §6.3)."""

        class HugeAtrLong:
            def evaluate(self, snapshot: IndicatorSnapshot) -> Any:
                from tachyon.strategy.signals import SignalReport

                return SignalReport(
                    symbol="RELIANCE",
                    signal=Signal.LONG,
                    reason=NoSignalReason.NONE,
                    ltp=2500.0,
                    session_vwap=2490.0,
                    ema=2495.0,
                    ema_period=21,
                    obi=0.5,
                    obi_weighted=0.5,
                    obi_threshold=0.3,
                    atr_5m=200.0,  # R = 300 > the 100 budget
                    ts_epoch=0.0,
                )

        harness.brain._generator = HugeAtrLong()  # type: ignore[assignment]  # noqa: SLF001
        harness.evaluate()
        await harness.settle()
        assert harness.placed == 0
        assert harness.brain.stats.build_rejections == 1

    async def test_a_second_signal_cannot_double_up_while_one_is_in_flight(
        self, tmp_path: Path
    ) -> None:
        """The window the risk gate cannot see: an order sent but not yet registered."""
        harness = _Harness(tmp_path, _clock())
        harness.force_long()
        harness.executor.delay = 0.05

        harness.evaluate()
        harness.evaluate()
        harness.evaluate()
        await harness.settle()

        assert harness.placed == 1
        assert harness.executor.max_concurrent == 1

    async def test_session_cap_blocks_further_entries(self, tmp_path: Path) -> None:
        harness = _Harness(
            tmp_path, _clock(), strategy=StrategySettings(max_signals_per_symbol_per_session=1)
        )
        harness.force_long()
        harness.evaluate()
        await harness.settle()
        assert harness.placed == 1

        harness.brain.positions.reset_session()
        harness.evaluate()
        await harness.settle()
        assert harness.placed == 1


class TestBrainMessageRouting:
    def _tick(self, seq: int, ltp: float = 2500.0, volume: int = 1000) -> Envelope:
        return Envelope(
            topic=tick_topic("RELIANCE").decode(),
            message=Tick(
                token="2885", ltp=ltp, volume=volume, ts_epoch=1_786_000_000.0 + seq, seq=seq
            ),
            received_mono=0.0,
        )

    def test_ticks_reach_the_aggregator(self, harness: _Harness) -> None:
        for seq in range(1, 6):
            harness.brain.on_message(self._tick(seq, ltp=2500.0 + seq, volume=1000 + seq * 10))
        assert harness.brain.stats.ticks == 5
        aggregator = harness.brain.aggregator("RELIANCE")
        assert aggregator is not None
        assert aggregator.tick_count == 5

    def test_depth_updates_reach_the_aggregator(self, harness: _Harness) -> None:
        book = OrderBook(
            token="2885",
            bid_price=(2499.95, 2499.90, 2499.85, 2499.80, 2499.75),
            bid_qty=(500, 400, 300, 200, 100),
            ask_price=(2500.05, 2500.10, 2500.15, 2500.20, 2500.25),
            ask_qty=(100, 90, 80, 70, 60),
            ts_epoch=1_786_000_000.0,
        )
        harness.brain.on_message(
            Envelope(topic=depth_topic("RELIANCE").decode(), message=book, received_mono=0.0)
        )
        assert harness.brain.stats.depth_updates == 1
        snapshot = harness.brain.snapshot("RELIANCE")
        assert snapshot is not None
        assert snapshot.obi > 0  # bid-heavy

    def test_heartbeats_are_counted_not_aggregated(self, harness: _Harness) -> None:
        harness.brain.on_message(
            Envelope(
                topic="FEED.HEARTBEAT",
                message=Heartbeat(ts_epoch=0.0, seq=1),
                received_mono=0.0,
            )
        )
        assert harness.brain.stats.heartbeats == 1
        assert harness.brain.stats.ticks == 0

    def test_unwatched_symbols_are_dropped(self, harness: _Harness) -> None:
        """Trading an instrument absent from the watchlist is forbidden (CLAUDE.md §8.1)."""
        harness.brain.on_message(
            Envelope(
                topic="TICK.YESBANK",
                message=Tick(token="9999", ltp=20.0, volume=1, ts_epoch=0.0, seq=1),
                received_mono=0.0,
            )
        )
        assert harness.brain.stats.unknown_symbols == 1
        assert harness.brain.stats.ticks == 0

    def test_a_handler_failure_does_not_stop_the_loop(self, harness: _Harness) -> None:
        class Exploding:
            def on_tick(self, tick: Tick) -> None:
                raise RuntimeError("kernel exploded")

            candle_count = 0

        harness.brain._symbols["RELIANCE"].aggregator = Exploding()  # type: ignore[assignment]  # noqa: SLF001
        harness.brain.on_message(self._tick(1))
        assert harness.brain.stats.handler_errors == 1

        # And the next message is still processed.
        harness.brain.on_message(
            Envelope(
                topic="FEED.HEARTBEAT",
                message=Heartbeat(ts_epoch=0.0, seq=1),
                received_mono=0.0,
            )
        )
        assert harness.brain.stats.heartbeats == 1

    def test_evaluation_cadence(self, tmp_path: Path) -> None:
        harness = _Harness(tmp_path, _clock(), strategy=StrategySettings(evaluate_every_n_ticks=5))
        for seq in range(1, 11):
            harness.brain.on_message(self._tick(seq, volume=1000 + seq))
        # Two cadence evaluations, plus one on the first candle opening.
        assert harness.brain.stats.evaluations >= 2


class TestBrainDoesNotBlock:
    async def test_ticks_keep_flowing_while_an_order_is_in_flight(self, tmp_path: Path) -> None:
        """A slow broker must not stall ingestion — a gap voids the session VWAP (§3.3)."""
        harness = _Harness(tmp_path, _clock())
        harness.force_long()
        harness.executor.delay = 0.25

        harness.evaluate()  # dispatches the entry as a task
        assert harness.brain._entry_tasks  # noqa: SLF001

        router = TestBrainMessageRouting()
        started = time.perf_counter()
        for seq in range(1, 201):
            harness.brain.on_message(router._tick(seq, volume=1000 + seq))
        elapsed = time.perf_counter() - started

        assert harness.brain.stats.ticks == 200
        assert elapsed < 0.2, f"200 ticks took {elapsed:.3f}s while an order was in flight"

        await harness.settle()
        assert harness.placed == 1

    async def test_the_event_loop_stays_responsive(self, tmp_path: Path) -> None:
        harness = _Harness(tmp_path, _clock())
        harness.force_long()
        harness.executor.delay = 0.2
        harness.evaluate()

        ticks = 0

        async def other_work() -> None:
            nonlocal ticks
            while ticks < 20:
                ticks += 1
                await asyncio.sleep(0.001)

        await asyncio.wait_for(other_work(), timeout=1.0)
        assert ticks == 20  # the loop was never monopolised
        await harness.settle()

    async def test_shutdown_awaits_entries_rather_than_cancelling_them(
        self, tmp_path: Path
    ) -> None:
        """Cancelling mid-placement produces exactly the unknown-outcome state §6.4 forbids."""
        harness = _Harness(tmp_path, _clock())
        harness.force_long()
        harness.executor.delay = 0.1
        harness.evaluate()

        await harness.brain.shutdown()
        assert harness.placed == 1


class TestBrainLifecycle:
    def test_position_close_books_pnl_cooldown_and_exposure(self, harness: _Harness) -> None:
        harness.brain.positions.record_entry("RELIANCE", 7)
        harness.brain.on_position_closed(
            "RELIANCE", realised_inr=Decimal("-42.50"), was_stop_out=True
        )
        assert harness.brain.pnl.realised == Decimal("-42.50")
        assert not harness.brain.positions.is_open("RELIANCE")
        assert not harness.brain.cooldown.may_enter("RELIANCE")

    def test_mark_to_market_can_trip_the_kill_switch(self, harness: _Harness) -> None:
        harness.brain.on_mark_to_market(Decimal("-501"))
        assert harness.brain.pnl.is_breached
        assert harness.brain.state is TradingState.LOCKED

    def test_reset_session_clears_per_day_state(self, harness: _Harness) -> None:
        harness.brain.cooldown.record_exit("RELIANCE", was_stop_out=True)
        harness.brain.on_message(
            Envelope(
                topic=tick_topic("RELIANCE").decode(),
                message=Tick(token="2885", ltp=2500.0, volume=1, ts_epoch=0.0, seq=1),
                received_mono=0.0,
            )
        )
        harness.brain.reset_session()
        assert harness.brain.stats.ticks == 0
        assert harness.brain.cooldown.may_enter("RELIANCE")

    def test_live_mode_without_a_client_is_refused(self) -> None:
        with pytest.raises(ValueError, match="LIVE mode requires"):
            StrategyBrain(settings=_settings(trading_mode=TradingMode.LIVE), client=None)

    def test_margin_provider_is_undefined_before_the_first_poll(self, harness: _Harness) -> None:
        """Not-yet-read must veto, never permit."""
        assert not harness.brain._current_margin().is_finite()  # noqa: SLF001


class TestExecutorAuthorisation:
    """`place_robo_order` is reachable without going through `open_position`."""

    async def test_refuses_an_unauthorised_decision(self, tmp_path: Path) -> None:
        from tachyon.risk.engine import RiskDecision

        clock = _clock()
        settings = _settings()
        machine = StateMachine(TradingState.ACTIVE, clock=clock)
        pnl = PnLTracker(machine, clock=clock)
        monitor = FeedMonitor(clock=clock)
        monitor.record()
        positions = PositionRegistry()
        risk = RiskEngine(machine, pnl, monitor, positions, settings=settings, clock=clock)
        builder = OrderBuilder(settings=settings, clock=clock)
        executor = RoboExecutor(
            client=None,
            builder=builder,
            risk=risk,
            positions=positions,
            journal=OrderJournal(tmp_path / "journal", clock=clock),
            settings=settings,
            mode=TradingMode.PAPER,
            clock=clock,
        )
        plan = builder.build(
            symbol="RELIANCE",
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )

        refused = RiskDecision(allowed=False, symbol="RELIANCE", at_ist=clock.now())
        report = await executor.place_robo_order(plan, refused)
        assert not report.placed
        assert report.rejected_reason == "NOT_AUTHORISED"

        wrong_symbol = RiskDecision(allowed=True, symbol="HDFCBANK", at_ist=clock.now())
        mismatch = await executor.place_robo_order(plan, wrong_symbol)
        assert not mismatch.placed
        assert mismatch.rejected_reason == "AUTHORISATION_MISMATCH"


class TestBrainBoot:
    """`boot()` ordering is load-bearing — CLAUDE.md §3, §9."""

    def _fresh(self, tmp_path: Path) -> StrategyBrain:
        clock = _clock()
        settings = _settings()
        brain = StrategyBrain(
            settings=settings,
            client=None,
            subscriber=None,
            state_machine=StateMachine(TradingState.ACTIVE, clock=clock),
            daily_lock=DailyLock(path=tmp_path / "daily_lock.txt", clock=clock),
            clock=clock,
        )
        brain._journal = OrderJournal(tmp_path / "journal", clock=clock)  # noqa: SLF001
        return brain

    async def test_paper_boot_is_tradeable_and_builds_aggregators(self, tmp_path: Path) -> None:
        brain = self._fresh(tmp_path)
        try:
            assert await brain.boot() is True
            assert brain.aggregator("RELIANCE") is not None
            assert brain.aggregator("HDFCBANK") is not None
            assert brain._watchdog.is_alive  # noqa: SLF001 - the 15:15 deadline must be armed
        finally:
            await brain.shutdown()

    async def test_a_failed_kernel_self_test_locks_the_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A process that computes wrong numbers must not reach the market (CLAUDE.md §3)."""
        import tachyon.strategy.brain as brain_module

        brain = self._fresh(tmp_path)
        monkeypatch.setattr(brain_module, "is_warm", lambda: False)
        monkeypatch.setattr(brain_module, "warmup", lambda: False)
        try:
            assert await brain.boot() is False
            assert brain.state is TradingState.LOCKED
            assert brain.aggregator("RELIANCE") is None
        finally:
            await brain.shutdown()

    async def test_an_engaged_daily_lock_boots_read_only(self, tmp_path: Path) -> None:
        brain = self._fresh(tmp_path)
        brain.pnl.trip("DAILY_LOSS_LIMIT", "forced")
        try:
            assert await brain.boot() is False
            assert brain.state is TradingState.LOCKED
            # The watchdog still runs: a locked session must still square off at 15:15.
            assert brain._watchdog.is_alive  # noqa: SLF001
        finally:
            await brain.shutdown()


class TestBrainEntryFailureModes:
    """What happens after an order goes wrong is where money is actually lost."""

    async def test_an_unknown_outcome_locks_the_session(self, tmp_path: Path) -> None:
        """A leg may be live at the broker with no local record (CLAUDE.md §6.4)."""
        harness = _Harness(tmp_path, _clock())
        harness.force_long()

        class UnknownOutcome:
            async def place_robo_order(self, plan: Any, decision: Any) -> ExecutionReport:
                return ExecutionReport(
                    symbol=plan.symbol,
                    side=plan.side,
                    at_ist="now",
                    plan=plan,
                    results=(
                        LegResult(
                            leg=plan.legs[0].leg,
                            order_tag=plan.legs[0].order_tag,
                            quantity=plan.legs[0].quantity,
                            accepted=False,
                            outcome_unknown=True,
                        ),
                    ),
                )

        harness.brain._executor = UnknownOutcome()  # type: ignore[assignment]  # noqa: SLF001
        harness.evaluate()
        await harness.settle()

        assert harness.brain.state is TradingState.LOCKED
        assert "RELIANCE" not in harness.brain._entries_in_flight  # noqa: SLF001

    async def test_an_unexpected_exception_locks_the_session(self, tmp_path: Path) -> None:
        harness = _Harness(tmp_path, _clock())
        harness.force_long()

        class Exploding:
            async def place_robo_order(self, plan: Any, decision: Any) -> ExecutionReport:
                raise RuntimeError("something we did not anticipate")

        harness.brain._executor = Exploding()  # type: ignore[assignment]  # noqa: SLF001
        harness.evaluate()
        await harness.settle()

        assert harness.brain.state is TradingState.LOCKED
        assert harness.brain.stats.entries_failed == 1
        # Released even on the failure path, so a retry is not blocked by a stale flag.
        assert "RELIANCE" not in harness.brain._entries_in_flight  # noqa: SLF001

    async def test_a_rejected_entry_does_not_lock(self, tmp_path: Path) -> None:
        """An ordinary broker rejection is a missed trade, not an emergency."""
        harness = _Harness(tmp_path, _clock())
        harness.force_long()

        class Rejecting:
            async def place_robo_order(self, plan: Any, decision: Any) -> ExecutionReport:
                return ExecutionReport(
                    symbol=plan.symbol,
                    side=plan.side,
                    at_ist="now",
                    plan=plan,
                    rejected_reason="INSUFFICIENT_FUNDS",
                )

        harness.brain._executor = Rejecting()  # type: ignore[assignment]  # noqa: SLF001
        harness.evaluate()
        await harness.settle()

        assert harness.brain.state is TradingState.ACTIVE
        assert harness.brain.stats.entries_failed == 1


class TestCooldownAccessors:
    def test_verdict_detail_when_free(self) -> None:
        manager = ReentryManager(clock=_clock())
        assert "no cooldown" in manager.check("RELIANCE").detail

    def test_cooldown_length_is_exposed(self) -> None:
        assert ReentryManager(clock=_clock()).cooldown == REENTRY_COOLDOWN

    def test_blocked_until_is_none_for_an_untraded_symbol(self) -> None:
        manager = ReentryManager(clock=_clock())
        assert manager.blocked_until("RELIANCE") is None
        manager.record_exit("RELIANCE")
        assert manager.blocked_until("RELIANCE") is not None


class TestBrainSessionBudget:
    """The dynamic session budget, wired end to end through the Brain — CLAUDE.md §1.3, §6.3.

    ``tests/unit/test_budget.py`` proves the arithmetic in isolation. These prove the number
    actually *reaches* the three components that enforce it: the P&L latch, the order sizer and
    the square-off watchdog's drawdown stop. A budget that resolves correctly and is then wired
    to nothing would pass every test in that file.
    """

    @staticmethod
    def _dynamic(tmp_path: Path, **capital: object) -> _Harness:
        defaults: dict[str, object] = {
            "session_budget_inr": Decimal("50000"),
            "max_daily_drawdown_pct": Decimal("2.0"),
            "per_trade_risk_pct": Decimal("0.4"),
        }
        defaults.update(capital)
        return _Harness(tmp_path, _clock(), capital=CapitalSettings(**defaults))  # type: ignore[arg-type]

    def test_unconfigured_capital_keeps_the_constitutional_limits(self, harness: _Harness) -> None:
        """The shipped default. §1's ₹500/₹100 remain in force until an operator opts in."""
        budget = harness.brain.budget
        assert budget.is_dynamic is False
        assert budget.daily_loss_limit == Decimal("500")
        assert budget.per_trade_risk == Decimal("100")
        assert harness.brain.pnl.limit == Decimal("500")

    def test_configured_capital_derives_both_limits(self, tmp_path: Path) -> None:
        brain = self._dynamic(tmp_path).brain
        assert brain.budget.is_dynamic is True
        assert brain.budget.daily_loss_limit == Decimal("1000.00")  # 50000 × 2.0%
        assert brain.budget.per_trade_risk == Decimal("200.00")  # 50000 × 0.4%

    def test_the_pnl_latch_enforces_the_derived_limit_not_the_constant(
        self, tmp_path: Path
    ) -> None:
        """The load-bearing one: a limit the tracker never received is decorative."""
        brain = self._dynamic(tmp_path).brain
        brain.pnl.book_realised(Decimal("-600"))
        assert brain.pnl.is_breached is False, "₹600 is inside a ₹1000 budget"
        brain.pnl.book_realised(Decimal("-400"))
        assert brain.pnl.is_breached is True, "₹1000 is the configured limit and must latch"

    def test_the_order_sizer_receives_the_derived_per_trade_risk(self, tmp_path: Path) -> None:
        """§6.3 sizes against the budget, so doubling it must double the shares."""
        builder = self._dynamic(tmp_path).brain._builder  # noqa: SLF001
        assert builder.per_trade_risk == Decimal("200.00")

        # Ample headroom, so the min() picks per_trade_risk: 200 / R=3 = 66 shares, floored.
        budget = builder.budget(headroom=Decimal("1000.00"))
        assert budget == Decimal("200.00")
        assert builder.size(risk_per_share=Decimal("3.00"), budget=budget).quantity == 66

    def test_headroom_still_binds_the_derived_budget(self, tmp_path: Path) -> None:
        """§6.3's ``min`` — a day that has spent most of its budget sizes against what remains,
        whichever way the per-trade number was derived."""
        builder = self._dynamic(tmp_path).brain._builder  # noqa: SLF001
        assert builder.budget(headroom=Decimal("75.00")) == Decimal("75.00")

    def test_the_watchdog_drawdown_stop_uses_the_derived_limit(self, tmp_path: Path) -> None:
        """§1.3's second enforcement point — the one that fires if the event loop wedges."""
        brain = self._dynamic(tmp_path).brain
        assert brain._watchdog.drawdown_limit == Decimal("1000.00")  # noqa: SLF001
        assert brain._watchdog.drawdown_tripped is False  # noqa: SLF001

    def test_pnl_frame_carries_the_budget_to_the_ui(self, tmp_path: Path) -> None:
        """§1.3 requirement 4 — the dashboard must show real headroom, not a hardcoded ₹500."""
        frame = self._dynamic(tmp_path).brain.pnl_frame()
        assert frame.limit == "1000.00"
        assert frame.capital == "50000"
        assert frame.drawdown_pct == "2.00"
        assert "50000" in frame.limit_source, "the operator must be able to see the derivation"

    def test_zero_capital_falls_back_to_the_constants(self, tmp_path: Path) -> None:
        """Failure resolves *downward* (§1.3). Everywhere else unknown state vetoes; here it
        picks the smaller of the two possible worlds rather than the larger."""
        brain = self._dynamic(tmp_path, session_budget_inr=Decimal("0")).brain
        assert brain.budget.is_dynamic is False
        assert brain.pnl.limit == Decimal("500")

    def test_negative_capital_is_refused_by_the_config_layer(self) -> None:
        """Defence in depth: the settings model rejects it before ``SessionBudget`` ever sees
        it. ``resolve`` still handles the case (test_budget.py) — two nets, not one."""
        with pytest.raises(ValidationError):
            CapitalSettings(session_budget_inr=Decimal("-1"))

    def test_a_sub_paise_budget_falls_back_rather_than_disabling_sizing(
        self, tmp_path: Path
    ) -> None:
        """0.4% of ₹1 rounds below a paisa. Zero risk budget would size every order to zero."""
        brain = self._dynamic(tmp_path, session_budget_inr=Decimal("1")).brain
        assert brain.budget.is_dynamic is False
        assert brain.budget.per_trade_risk == Decimal("100")


class TestBrainTelemetryAndFills:
    """Phase 10 wiring — the state spine and the fill loop, CLAUDE.md §2.1, §7.3."""

    def test_state_and_pnl_frames_carry_the_whole_view(self, harness: _Harness) -> None:
        """Whole-state, not a delta: the UI conflates and will miss frames (§2.1)."""
        harness.brain.positions.record_entry("RELIANCE", 7)
        harness.brain.pnl.book_realised(Decimal("-42.50"), Decimal("12.30"))

        state = harness.brain.state_frame()
        assert state.state == "ACTIVE"
        assert state.mode == "PAPER"
        assert state.open_symbols == ("RELIANCE",)

        pnl = harness.brain.pnl_frame()
        assert pnl.realised == "-42.50", "money crosses as a string, never a float"
        assert pnl.charges == "12.30"
        assert pnl.limit == "500"

    def test_publish_telemetry_sends_state_and_pnl(self, harness: _Harness) -> None:
        harness.brain.publish_telemetry()
        assert harness.publisher.topics() == ["STATE.SESSION", "PNL.SESSION"]

    def test_constructing_a_brain_binds_nothing(self, tmp_path: Path) -> None:
        """Construction must not claim an OS resource.

        Two brains in one process is normal — in a test, and in any supervisor that builds one
        before deciding to start it. A constructor that binds a port makes the second one fight
        the first, and leaves a ZeroMQ context that never terminates.
        """
        real = StatePublisher(settings=_settings(), clock=_clock())
        assert not real.is_bound

    def test_telemetry_never_raises_when_it_cannot_bind(self) -> None:
        """Losing the dashboard must not stop the Brain."""
        import tachyon.strategy.telemetry as telemetry_module

        def refuse(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError("address already in use")

        publisher = StatePublisher(settings=_settings(), clock=_clock())
        original = telemetry_module.Publisher
        telemetry_module.Publisher = refuse  # type: ignore[misc, assignment]
        try:
            publisher.publish_risk("VETO", "RELIANCE", "FEED_STALE")  # must not raise
            publisher.publish_risk("VETO", "RELIANCE", "FEED_STALE")  # and stays disabled
        finally:
            telemetry_module.Publisher = original  # type: ignore[misc]

        assert publisher.errors == 2
        assert not publisher.is_bound

    def test_a_fill_closure_books_pnl_and_starts_the_cooldown(self, harness: _Harness) -> None:
        """The wire that was missing until Phase 10."""
        harness.brain.positions.record_entry("RELIANCE", 10)
        harness.brain.fills.ingest_raw(_fill("X1", "BUY", "2500.00"))
        harness.brain.fills.ingest_raw(_fill("X2", "SELL", "2487.40", order_type="STOPLOSS_LIMIT"))

        assert harness.brain.pnl.realised == Decimal("-126.00")
        assert harness.brain.pnl.charges > 0
        assert not harness.brain.positions.is_open("RELIANCE")
        assert not harness.brain.cooldown.may_enter("RELIANCE"), "a stop-out starts the cooldown"

    def test_fills_are_forwarded_to_the_ui(self, harness: _Harness) -> None:
        harness.brain.fills.ingest_raw(_fill("Y1", "BUY", "2500.00"))
        assert "FILL.RELIANCE" in harness.publisher.topics()

    def test_a_duplicate_fill_does_not_double_the_pnl(self, harness: _Harness) -> None:
        """The webhook and the poller deliver the same event; P&L must move once."""
        harness.brain.positions.record_entry("RELIANCE", 10)
        for _ in range(3):
            harness.brain.fills.ingest_raw(_fill("Z1", "BUY", "2500.00"))
            harness.brain.fills.ingest_raw(_fill("Z2", "SELL", "2510.00"))
        assert harness.brain.pnl.realised == Decimal("100.00")

    def test_an_external_lock_is_adopted(self, harness: _Harness) -> None:
        """The UI's panic button reaches the state machine within one liveness tick (§7.3)."""
        harness.brain.pnl.daily_lock.engage("OPERATOR_PANIC")
        assert harness.brain.state is TradingState.ACTIVE

        harness.brain._honour_external_lock()  # noqa: SLF001
        assert harness.brain.state is TradingState.LOCKED

    def test_a_vanished_lock_never_unlocks(self, harness: _Harness) -> None:
        """LOCKED is absorbing. A system that can un-halt itself is one that will (§4)."""
        harness.brain.pnl.daily_lock.engage("OPERATOR_PANIC")
        harness.brain._honour_external_lock()  # noqa: SLF001
        harness.brain.pnl.daily_lock.path.unlink()

        harness.brain._honour_external_lock()  # noqa: SLF001
        assert harness.brain.state is TradingState.LOCKED


def _fill(
    order_id: str, side: str, price: str, *, order_type: str = "LIMIT", filled: int = 10
) -> dict[str, Any]:
    return {
        "orderid": order_id,
        "symboltoken": "2885",
        "tradingsymbol": "RELIANCE-EQ",
        "transactiontype": side,
        "orderstatus": "complete",
        "quantity": filled,
        "filledshares": filled,
        "averageprice": price,
        "ordertype": order_type,
        # Our client tag. Without it the listener treats the fill as rogue and locks the day
        # (CLAUDE.md §9) — which is exactly the point of the tag.
        "ordertag": "TCHYN-20260810-0001",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Telegram operator alerts (utils/telegram_alerts.py) — the Brain's wiring
# ──────────────────────────────────────────────────────────────────────────────


def _alerting(harness: _Harness) -> tuple[TelegramAlerter, list[str]]:
    """Swap in a real alerter over a mock transport, so the wiring is exercised end to end.

    A hand-written double would prove the Brain calls *something*; this proves the message
    that would reach the operator's phone actually contains what it needs to.
    """
    sent: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(str(json.loads(request.content)["text"]))
        return httpx.Response(200, json={"ok": True})

    alerter = TelegramAlerter(
        bot_token="123:FAKE",
        chat_id="42",
        transport=httpx.MockTransport(record),
        clock=harness.clock,
        start=False,
    )
    harness.brain._alerts = alerter  # noqa: SLF001
    return alerter, sent


class TestTelegramAlertWiring:
    """CLAUDE.md §1.1, §6.1, §7.4 — the three events an operator must learn about."""

    async def test_an_entry_alerts_with_its_geometry(self, harness: _Harness) -> None:
        alerter, sent = _alerting(harness)
        harness.force_long()
        harness.evaluate()
        await harness.settle()
        alerter.drain_for_test()

        assert len(sent) == 1
        assert "RELIANCE" in sent[0]
        assert "LONG" in sent[0]
        assert "entry" in sent[0] and "stop" in sent[0]
        assert "[PAPER]" in sent[0], "a simulated placement must not read as a real one"

    async def test_a_veto_alerts_nothing(self, harness: _Harness) -> None:
        """Alerts mark what happened, not what was considered — vetoes go to the veto CSV."""
        alerter, sent = _alerting(harness)
        harness.force_long()
        harness.machine.transition_to(TradingState.NO_NEW_ENTRIES, reason="test")
        harness.evaluate()
        await harness.settle()
        alerter.drain_for_test()
        assert sent == []

    def test_a_stop_out_is_alerted_as_a_stop_out(self, harness: _Harness) -> None:
        alerter, sent = _alerting(harness)
        harness.brain._on_fill_closed(  # noqa: SLF001
            "RELIANCE",
            Decimal("-100.00"),
            Decimal("20.00"),
            True,  # noqa: FBT003 - fixed positional callback signature
            now_ist(harness.clock),
        )
        alerter.drain_for_test()
        assert len(sent) == 1
        assert "STOP-LOSS" in sent[0]
        assert "-Rs.120.00" in sent[0], "net of charges — the number the limit is enforced on"

    def test_a_target_close_is_not_alerted_as_a_stop_out(self, harness: _Harness) -> None:
        alerter, sent = _alerting(harness)
        harness.brain._on_fill_closed(  # noqa: SLF001
            "RELIANCE",
            Decimal("150.00"),
            Decimal("20.00"),
            False,  # noqa: FBT003 - fixed positional callback signature
            now_ist(harness.clock),
        )
        alerter.drain_for_test()
        assert "TARGET" in sent[0]
        assert "STOP-LOSS" not in sent[0]

    def test_an_alerter_that_raises_never_reaches_the_booking_path(self, harness: _Harness) -> None:
        """§5.1's asymmetry: a broken advisor must not become an exception on the P&L path."""

        class _Exploding:
            def __getattr__(self, _name: str) -> object:
                def boom(**_kw: object) -> bool:
                    raise RuntimeError("telegram is down")

                return boom

        harness.brain._alerts = _Exploding()  # type: ignore[assignment]  # noqa: SLF001
        with pytest.raises(RuntimeError):
            # Confirms the double really does raise, so the assertions below mean something.
            harness.brain._alerts.position_closed()  # noqa: SLF001

        harness.brain._on_fill_closed(  # noqa: SLF001
            "RELIANCE",
            Decimal("-100.00"),
            Decimal("20.00"),
            True,  # noqa: FBT003 - fixed positional callback signature
            now_ist(harness.clock),
        )

        # Booked despite the alerter exploding — the guard is what makes this hold.
        assert harness.brain.pnl.total == Decimal("-120.00")
        assert harness.brain.cooldown.may_enter("RELIANCE") is False

    async def test_an_alerter_that_raises_never_breaks_an_entry(self, harness: _Harness) -> None:
        class _Exploding:
            def __getattr__(self, _name: str) -> object:
                def boom(**_kw: object) -> bool:
                    raise RuntimeError("telegram is down")

                return boom

        harness.brain._alerts = _Exploding()  # type: ignore[assignment]  # noqa: SLF001
        harness.force_long()
        harness.evaluate()
        await harness.settle()

        assert harness.placed == 1
        assert harness.brain.stats.entries_placed == 1
        assert "RELIANCE" not in harness.brain._entries_in_flight  # noqa: SLF001

    def test_the_square_off_wrapper_re_raises_so_the_watchdog_retries(
        self, harness: _Harness
    ) -> None:
        """The watchdog knows to retry *because* the action raises (CLAUDE.md §1.1, §6.5)."""
        alerter, sent = _alerting(harness)
        attempts = 0

        def failing() -> None:
            nonlocal attempts
            attempts += 1
            raise NotFlatError("1 working order")

        action = harness.brain._alerting_square_off(failing)  # noqa: SLF001

        for _ in range(3):
            with pytest.raises(NotFlatError):
                action()
        assert attempts == 3, "the wrapper must not absorb the failure"

        alerter.drain_for_test()
        assert len(sent) == 2, "fired once + failed once, not once per retry"
        assert "15:15" in sent[0]
        assert "Check the broker terminal" in sent[1]

    def test_the_square_off_wrapper_alerts_on_eventual_success(self, harness: _Harness) -> None:
        alerter, sent = _alerting(harness)
        calls = 0

        def flaky() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise NotFlatError("broker timeout")

        action = harness.brain._alerting_square_off(flaky)  # noqa: SLF001
        with pytest.raises(NotFlatError):
            action()
        action()

        alerter.drain_for_test()
        assert len(sent) == 3
        assert "flat" in sent[2], "the operator must learn the retry worked"


class TestKillSwitchAndRogueFillAlerts:
    """CLAUDE.md §1.2, §7.4 — the two lockdowns the operator must hear about."""

    def test_the_loss_limit_alerts_once(self, harness: _Harness) -> None:
        alerter, sent = _alerting(harness)
        harness.brain.pnl.book_realised(Decimal("-500"))
        alerter.drain_for_test()

        assert len(sent) == 1, "the latch is idempotent, so exactly one buzz"
        assert "TRADING HALTED" in sent[0]
        assert "DAILY_LOSS_LIMIT" in sent[0]
        assert harness.brain.state is TradingState.LOCKED

    def test_a_second_breach_does_not_re_alert(self, harness: _Harness) -> None:
        alerter, sent = _alerting(harness)
        harness.brain.pnl.book_realised(Decimal("-500"))
        harness.brain.pnl.book_realised(Decimal("-500"))
        harness.brain.pnl.update_floating(Decimal("-900"))
        alerter.drain_for_test()
        assert len(sent) == 1

    def test_the_watchdog_drawdown_path_alerts_too(self, harness: _Harness) -> None:
        """The §1.3 hard-stop rides on the watchdog thread, not the P&L update path."""
        alerter, sent = _alerting(harness)
        harness.brain._on_drawdown_breach(Decimal("-501"))  # noqa: SLF001
        alerter.drain_for_test()
        assert len(sent) == 1
        assert "DAILY_DRAWDOWN_BREACHED" in sent[0]

    def test_an_unwritable_lock_file_warns_against_restarting(
        self, harness: _Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§1.2: this session is locked, a restart would not be. The operator must know."""
        alerter, sent = _alerting(harness)

        def explode(*_a: object, **_kw: object) -> None:
            raise OSError("read-only file system")

        # DailyLock is a frozen dataclass, so the patch goes on the class, not the instance.
        monkeypatch.setattr(type(harness.brain.pnl.daily_lock), "engage", explode)
        harness.brain.pnl.book_realised(Decimal("-500"))
        alerter.drain_for_test()

        assert len(sent) == 1
        assert "RESTART WOULD NOT BE" in sent[0]
        assert harness.brain.state is TradingState.LOCKED, "the latch still engages"

    def test_a_rogue_fill_alerts_twice_and_locks(self, harness: _Harness) -> None:
        """Specifics first, then the kill switch. The only event worth two buzzes."""
        alerter, sent = _alerting(harness)
        update = OrderUpdate.from_broker(
            _fill("ROGUE-1", "BUY", "1402.50"),
            lambda _t, _s: "RELIANCE",
        )
        harness.brain._on_rogue_fill(update)  # noqa: SLF001
        alerter.drain_for_test()

        assert len(sent) == 2
        assert "ROGUE FILL" in sent[0]
        assert "ROGUE-1" in sent[0]
        assert "NOT booked" in sent[0]
        assert "KILL SWITCH" in sent[1]
        assert "OPEN THE BROKER TERMINAL NOW" in sent[0]
        assert harness.brain.state is TradingState.LOCKED

    def test_a_non_pnl_lockdown_alerts_once(self, harness: _Harness) -> None:
        """`_lock` covers ENTRY_OUTCOME_UNKNOWN, which used to halt the session silently."""
        alerter, sent = _alerting(harness)
        harness.brain._lock("ENTRY_OUTCOME_UNKNOWN")  # noqa: SLF001
        alerter.drain_for_test()

        assert len(sent) == 1
        assert "SESSION LOCKED" in sent[0]
        assert "ENTRY_OUTCOME_UNKNOWN" in sent[0]

    def test_locking_an_already_locked_session_does_not_re_alert(self, harness: _Harness) -> None:
        alerter, sent = _alerting(harness)
        harness.brain._lock("FIRST")  # noqa: SLF001
        harness.brain._lock("SECOND")  # noqa: SLF001
        alerter.drain_for_test()
        assert len(sent) == 1
        assert "FIRST" in sent[0]

    def test_a_broken_alerter_never_stops_the_kill_switch(self, harness: _Harness) -> None:
        """§1.2: a kill switch that can fail is not a kill switch."""

        class _Exploding:
            def __getattr__(self, _name: str) -> object:
                def boom(**_kw: object) -> bool:
                    raise RuntimeError("telegram is down")

                return boom

        harness.brain._alerts = _Exploding()  # type: ignore[assignment]  # noqa: SLF001
        harness.brain.pnl.book_realised(Decimal("-500"))

        assert harness.brain.state is TradingState.LOCKED
        assert harness.brain.pnl.daily_lock.is_engaged()
