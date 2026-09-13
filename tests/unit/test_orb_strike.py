"""Tests for the 5-Minute Opening Range Breakout (ORB) strategy.

The strategy is a state machine on top of the live tick stream. These tests exercise
each transition deterministically by feeding ticks at known IST moments — no monkey-
patching of the system clock, no real :class:`datetime.now` calls.

The tick timestamp helper builds an IST :class:`datetime` and converts to epoch, so
the strategy's own ``tick.ts_ist.time()`` derives the moment from the same epoch the
test controls.
"""

from __future__ import annotations

import math
from datetime import datetime, time

import pytest
from zoneinfo import ZoneInfo

from tachyon.core.clock import IST
from tachyon.ipc.schemas import Depth5Price, Depth5Qty, OrderBook, Tick
from tachyon.strategies.orb_strike import (
    GOLDEN_WINDOW_END_IST,
    MARKETABLE_LIMIT_SLIPPAGE_BPS,
    OBI_THRESHOLD,
    OrbSide,
    OrbSignal,
    OrbState,
    OrbStrikeEngine,
    RVOL_THRESHOLD,
)

IST_TZ = ZoneInfo("Asia/Kolkata")


def _ist_epoch(hour: int, minute: int, second: int = 0) -> float:
    """Build an epoch timestamp in IST for a chosen wall-clock time.

    Pinning the date to a session-typical day keeps the timestamps out of any DST
    transition (IST is fixed at +05:30 with no DST). 2026-03-15 is well outside any
    transition in the tzdata database.
    """
    dt = datetime(2026, 3, 15, hour, minute, second, tzinfo=IST_TZ)
    return dt.timestamp()


def _tick(ltp: float, *, hour: int, minute: int, second: int = 0, seq: int = 1) -> Tick:
    return Tick(
        token="2885",
        ltp=ltp,
        volume=0,
        ts_epoch=_ist_epoch(hour, minute, second),
        seq=seq,
    )


def _book(bid_qty: int = 100, ask_qty: int = 100) -> OrderBook:
    return OrderBook(
        token="2885",
        bid_price=(100.0, 99.0, 98.0, 97.0, 96.0),
        bid_qty=(bid_qty, 0, 0, 0, 0),
        ask_price=(101.0, 102.0, 103.0, 104.0, 105.0),
        ask_qty=(ask_qty, 0, 0, 0, 0),
        ts_epoch=0.0,
    )


def _configure_for_fire(
    engine: OrbStrikeEngine,
    *,
    current_bar_volume: float = 5_000.0,
    baseline_volume: float = 1_000.0,
    atr_5m: float = 12.0,
) -> None:
    engine.set_context(
        current_bar_volume=current_bar_volume,
        baseline_volume=baseline_volume,
        atr_5m=atr_5m,
    )


# ──────────────────────────────────────────────────────────────────────────────
# State machine
# ──────────────────────────────────────────────────────────────────────────────


class TestStateMachine:
    def test_starts_in_pre_open(self) -> None:
        engine = OrbStrikeEngine("RELIANCE")
        assert engine.state is OrbState.PRE_OPEN
        assert engine.fired is False
        assert math.isnan(engine.orb_high)
        assert math.isnan(engine.orb_low)
        assert math.isnan(engine.range_width)

    def test_first_tick_in_building_window_advances_state(self) -> None:
        engine = OrbStrikeEngine("RELIANCE")
        engine.on_tick(_tick(100.0, hour=9, minute=15))
        assert engine.state is OrbState.BUILDING

    def test_building_window_accumulates_high_and_low(self) -> None:
        engine = OrbStrikeEngine("RELIANCE")
        for ltp, minute in ((99.0, 15), (101.0, 16), (100.5, 17), (100.8, 18), (100.2, 19)):
            engine.on_tick(_tick(ltp, hour=9, minute=minute, seq=minute))
        assert engine.state is OrbState.BUILDING
        assert engine.orb_high == pytest.approx(101.0)
        assert engine.orb_low == pytest.approx(99.0)

    def test_tick_at_0920_locks_the_range(self) -> None:
        engine = OrbStrikeEngine("RELIANCE")
        for ltp, minute in ((99.0, 15), (101.0, 16), (100.5, 17)):
            engine.on_tick(_tick(ltp, hour=9, minute=minute, seq=minute))
        engine.on_tick(_tick(100.7, hour=9, minute=20))
        assert engine.state is OrbState.LOCKED
        assert engine.orb_high == pytest.approx(101.0)
        assert engine.orb_low == pytest.approx(99.0)
        assert engine.range_width == pytest.approx(2.0)

    def test_lock_is_immutable(self) -> None:
        """Once locked, orb_high/orb_low must not be affected by later ticks."""
        engine = OrbStrikeEngine("RELIANCE")
        for ltp, minute in ((99.0, 15), (101.0, 16)):
            engine.on_tick(_tick(ltp, hour=9, minute=minute, seq=minute))
        engine.on_tick(_tick(100.7, hour=9, minute=20))
        # Tick that would otherwise expand the range must not change it.
        engine.on_tick(_tick(110.0, hour=9, minute=21, seq=4))
        assert engine.orb_high == pytest.approx(101.0)
        assert engine.orb_low == pytest.approx(99.0)

    def test_window_at_0945_expires_the_engine(self) -> None:
        engine = OrbStrikeEngine("RELIANCE")
        for ltp, minute in ((99.0, 15), (101.0, 16)):
            engine.on_tick(_tick(ltp, hour=9, minute=minute, seq=minute))
        engine.on_tick(_tick(100.7, hour=9, minute=20))
        engine.on_tick(_tick(100.0, hour=9, minute=45))
        assert engine.state is OrbState.EXPIRED

    def test_empty_building_window_expires(self) -> None:
        """A symbol that never ticks before 09:20 has no range to break out of."""
        engine = OrbStrikeEngine("RELIANCE")
        engine.on_tick(_tick(100.0, hour=9, minute=20))
        assert engine.state is OrbState.EXPIRED

    def test_reset_session_returns_to_pre_open(self) -> None:
        engine = OrbStrikeEngine("RELIANCE")
        engine.on_tick(_tick(100.0, hour=9, minute=15))
        engine.on_tick(_tick(101.0, hour=9, minute=20))
        engine.reset_session()
        assert engine.state is OrbState.PRE_OPEN
        assert engine.fired is False
        assert engine.last_signal is None


# ──────────────────────────────────────────────────────────────────────────────
# Trigger logic
# ──────────────────────────────────────────────────────────────────────────────


def _engine_with_built_range() -> OrbStrikeEngine:
    """A standard engine: range 99-101, locked at 09:20, no fire yet."""
    engine = OrbStrikeEngine("RELIANCE")
    for ltp, minute in ((99.0, 15), (101.0, 16), (100.5, 17), (100.8, 18), (100.2, 19)):
        engine.on_tick(_tick(ltp, hour=9, minute=minute, seq=minute))
    engine.on_tick(_tick(100.7, hour=9, minute=20, seq=20))
    assert engine.state is OrbState.LOCKED
    return engine


class TestLongTrigger:
    def test_fires_when_ltp_crosses_above_with_all_confirmations(self) -> None:
        engine = _engine_with_built_range()
        _configure_for_fire(engine, baseline_volume=1_000.0)  # RVOL = 5.0
        signal = engine.on_tick(
            _tick(101.5, hour=9, minute=21, seq=21),
            book=_book(bid_qty=400, ask_qty=100),  # OBI ≈ +0.60
        )
        assert signal is not None
        assert signal.is_long
        assert signal.orb_high == pytest.approx(101.0)
        assert signal.orb_low == pytest.approx(99.0)
        assert signal.ltp == pytest.approx(101.5)
        assert signal.atr_5m == pytest.approx(12.0)
        assert signal.obi >= OBI_THRESHOLD
        assert signal.rvol >= RVOL_THRESHOLD

    def test_no_fire_when_ltp_above_range_but_obi_weak(self) -> None:
        engine = _engine_with_built_range()
        _configure_for_fire(engine)
        # OBI ≈ 0.0: balanced book, no directional confirmation.
        signal = engine.on_tick(
            _tick(101.5, hour=9, minute=21, seq=21),
            book=_book(bid_qty=100, ask_qty=100),
        )
        assert signal is None

    def test_no_fire_when_obi_strong_but_rvol_low(self) -> None:
        engine = _engine_with_built_range()
        _configure_for_fire(engine, current_bar_volume=1_000.0, baseline_volume=1_000.0)
        signal = engine.on_tick(
            _tick(101.5, hour=9, minute=21, seq=21),
            book=_book(bid_qty=400, ask_qty=100),
        )
        assert signal is None  # RVOL = 1.0 < 2.5

    def test_no_fire_when_ltp_inside_the_range(self) -> None:
        engine = _engine_with_built_range()
        _configure_for_fire(engine)
        signal = engine.on_tick(
            _tick(100.5, hour=9, minute=21, seq=21),
            book=_book(bid_qty=400, ask_qty=100),
        )
        assert signal is None

    def test_no_fire_when_book_is_crossed(self) -> None:
        engine = _engine_with_built_range()
        _configure_for_fire(engine)
        # No book at all: the strategy refuses to fire on a phantom ask.
        signal = engine.on_tick(_tick(101.5, hour=9, minute=21, seq=21), book=None)
        assert signal is None


class TestShortTrigger:
    def test_fires_when_ltp_crosses_below_with_all_confirmations(self) -> None:
        engine = _engine_with_built_range()
        _configure_for_fire(engine)
        signal = engine.on_tick(
            _tick(98.5, hour=9, minute=21, seq=21),
            book=_book(bid_qty=100, ask_qty=400),  # OBI ≈ -0.60
        )
        assert signal is not None
        assert signal.is_short
        assert signal.obi <= -OBI_THRESHOLD

    def test_no_fire_when_obi_against_direction(self) -> None:
        engine = _engine_with_built_range()
        _configure_for_fire(engine)
        signal = engine.on_tick(
            _tick(98.5, hour=9, minute=21, seq=21),
            book=_book(bid_qty=400, ask_qty=100),  # bid-heavy, against a short
        )
        assert signal is None


class TestLatching:
    def test_first_signal_locks_out_subsequent_ones(self) -> None:
        engine = _engine_with_built_range()
        _configure_for_fire(engine)
        first = engine.on_tick(
            _tick(101.5, hour=9, minute=21, seq=21),
            book=_book(bid_qty=400, ask_qty=100),
        )
        assert first is not None
        assert engine.fired is True
        assert engine.state is OrbState.EXPIRED

        # A second crossover, even with perfect confirmations, must not fire.
        second = engine.on_tick(
            _tick(102.0, hour=9, minute=22, seq=22),
            book=_book(bid_qty=400, ask_qty=100),
        )
        assert second is None
        # Last signal still points at the first fire.
        assert engine.last_signal is first

    def test_latch_prevents_opposite_side_fire(self) -> None:
        """A long fire must not be followed by a short fire on the next bar."""
        engine = _engine_with_built_range()
        _configure_for_fire(engine)
        engine.on_tick(
            _tick(101.5, hour=9, minute=21, seq=21),
            book=_book(bid_qty=400, ask_qty=100),
        )
        # Same tick, opposite direction: the latch must win.
        _ = engine.on_tick(
            _tick(98.5, hour=9, minute=22, seq=22),
            book=_book(bid_qty=100, ask_qty=400),
        )
        assert engine.last_signal is not None
        assert engine.last_signal.is_long


class TestSignalPrice:
    def test_long_price_is_above_best_ask_by_slippage(self) -> None:
        engine = _engine_with_built_range()
        _configure_for_fire(engine)
        book = _book(bid_qty=400, ask_qty=100)
        best_ask = book.ask_price[0]
        signal = engine.on_tick(
            _tick(101.5, hour=9, minute=21, seq=21), book=book
        )
        assert signal is not None
        expected = float(best_ask) * (1.0 + MARKETABLE_LIMIT_SLIPPAGE_BPS / 10_000.0)
        assert float(signal.price) == pytest.approx(expected, rel=1e-9)

    def test_short_price_is_below_best_bid_by_slippage(self) -> None:
        engine = _engine_with_built_range()
        _configure_for_fire(engine)
        book = _book(bid_qty=100, ask_qty=400)
        best_bid = book.bid_price[0]
        signal = engine.on_tick(
            _tick(98.5, hour=9, minute=21, seq=21), book=book
        )
        assert signal is not None
        expected = float(best_bid) * (1.0 - MARKETABLE_LIMIT_SLIPPAGE_BPS / 10_000.0)
        assert float(signal.price) == pytest.approx(expected, rel=1e-9)


class TestSignalIntegrity:
    def test_signal_rejects_nan_atr(self) -> None:
        with pytest.raises(ValueError, match="atr"):
            OrbSignal(
                symbol="RELIANCE",
                side=OrbSide.LONG,
                price=__import__("decimal").Decimal("100.0"),
                orb_high=101.0,
                orb_low=99.0,
                ltp=101.5,
                obi=0.5,
                rvol=3.0,
                atr_5m=math.nan,
                ts_epoch=0.0,
            )

    def test_no_fire_without_valid_atr(self) -> None:
        engine = _engine_with_built_range()
        engine.set_context(current_bar_volume=5_000.0, baseline_volume=1_000.0, atr_5m=math.nan)
        signal = engine.on_tick(
            _tick(101.5, hour=9, minute=21, seq=21),
            book=_book(bid_qty=400, ask_qty=100),
        )
        assert signal is None

    def test_no_fire_when_rvol_kernel_returns_zero(self) -> None:
        """A zero baseline yields RVOL=0; the strategy must treat that as a veto."""
        engine = _engine_with_built_range()
        engine.set_context(current_bar_volume=5_000.0, baseline_volume=0.0, atr_5m=12.0)
        signal = engine.on_tick(
            _tick(101.5, hour=9, minute=21, seq=21),
            book=_book(bid_qty=400, ask_qty=100),
        )
        assert signal is None


class TestModuleConstants:
    def test_golden_window_end_is_0945(self) -> None:
        assert GOLDEN_WINDOW_END_IST == time(9, 45)

    def test_thresholds_match_spec(self) -> None:
        assert OBI_THRESHOLD == pytest.approx(0.35)
        assert RVOL_THRESHOLD == pytest.approx(2.5)
        assert MARKETABLE_LIMIT_SLIPPAGE_BPS == pytest.approx(5.0)
