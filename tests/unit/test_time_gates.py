"""Tests for the time-of-day market phase machine.

The phase table is the most safety-critical mapping in the engine: a wrong answer
on the live tick path is a real-money wrong order. The tests below walk every
boundary moment and a handful of mid-phase ticks, with the clock pinned to IST.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from tachyon.strategies.time_gates import (
    MarketPhase,
    get_market_phase,
    is_momentum_blocked,
)

IST = ZoneInfo("Asia/Kolkata")


def _ist_epoch(hour: int, minute: int, second: int = 0) -> float:
    """IST-anchored epoch. 2026-03-15 is far from any DST flip for IST (which has none)."""
    return datetime(2026, 3, 15, hour, minute, second, tzinfo=IST).timestamp()


class TestPhaseBoundaries:
    def test_pre_market_is_pre_open(self) -> None:
        assert get_market_phase(_ist_epoch(0, 0)) is MarketPhase.PRE_OPEN
        assert get_market_phase(_ist_epoch(9, 0)) is MarketPhase.PRE_OPEN
        assert get_market_phase(_ist_epoch(9, 14, 59)) is MarketPhase.PRE_OPEN

    def test_orb_build_opens_at_0915(self) -> None:
        assert get_market_phase(_ist_epoch(9, 15)) is MarketPhase.ORB_BUILD
        assert get_market_phase(_ist_epoch(9, 19, 59)) is MarketPhase.ORB_BUILD

    def test_golden_window_opens_at_0920(self) -> None:
        assert get_market_phase(_ist_epoch(9, 20)) is MarketPhase.GOLDEN_WINDOW
        assert get_market_phase(_ist_epoch(9, 44, 59)) is MarketPhase.GOLDEN_WINDOW

    def test_midday_chop_opens_at_0945(self) -> None:
        assert get_market_phase(_ist_epoch(9, 45)) is MarketPhase.MIDDAY_CHOP
        assert get_market_phase(_ist_epoch(13, 29, 59)) is MarketPhase.MIDDAY_CHOP

    def test_afternoon_trend_opens_at_1330(self) -> None:
        assert get_market_phase(_ist_epoch(13, 30)) is MarketPhase.AFTERNOON_TREND
        assert get_market_phase(_ist_epoch(14, 59, 59)) is MarketPhase.AFTERNOON_TREND

    def test_square_off_opens_at_1500(self) -> None:
        assert get_market_phase(_ist_epoch(15, 0)) is MarketPhase.SQUARE_OFF
        assert get_market_phase(_ist_epoch(15, 29, 59)) is MarketPhase.SQUARE_OFF

    def test_closed_after_1530(self) -> None:
        assert get_market_phase(_ist_epoch(15, 30)) is MarketPhase.CLOSED
        assert get_market_phase(_ist_epoch(23, 59, 59)) is MarketPhase.CLOSED


class TestPhaseMidpoints:
    """Sanity-check that mid-phase ticks resolve correctly, not just the boundaries."""

    def test_midday_chop_midpoints(self) -> None:
        for hour, minute in ((10, 30), (11, 45), (12, 0), (13, 0)):
            assert get_market_phase(_ist_epoch(hour, minute)) is MarketPhase.MIDDAY_CHOP

    def test_afternoon_trend_midpoints(self) -> None:
        for hour, minute in ((14, 0), (14, 15), (14, 30)):
            assert get_market_phase(_ist_epoch(hour, minute)) is MarketPhase.AFTERNOON_TREND


class TestPhaseIntEnum:
    def test_phases_have_expected_values(self) -> None:
        """The integer values are themselves meaningful: PRE_OPEN sorts before trading."""
        assert MarketPhase.PRE_OPEN < MarketPhase.ORB_BUILD
        assert MarketPhase.ORB_BUILD < MarketPhase.GOLDEN_WINDOW
        assert MarketPhase.GOLDEN_WINDOW < MarketPhase.MIDDAY_CHOP
        assert MarketPhase.MIDDAY_CHOP < MarketPhase.AFTERNOON_TREND
        assert MarketPhase.AFTERNOON_TREND < MarketPhase.SQUARE_OFF
        assert MarketPhase.SQUARE_OFF < MarketPhase.CLOSED


class TestMomentumBlockRule:
    def test_momentum_blocked_in_midday_chop(self) -> None:
        assert is_momentum_blocked(MarketPhase.MIDDAY_CHOP) is True

    def test_momentum_blocked_in_square_off(self) -> None:
        assert is_momentum_blocked(MarketPhase.SQUARE_OFF) is True

    def test_momentum_allowed_in_orb_build(self) -> None:
        assert is_momentum_blocked(MarketPhase.ORB_BUILD) is False

    def test_momentum_allowed_in_golden_window(self) -> None:
        assert is_momentum_blocked(MarketPhase.GOLDEN_WINDOW) is False

    def test_momentum_allowed_in_afternoon_trend(self) -> None:
        assert is_momentum_blocked(MarketPhase.AFTERNOON_TREND) is False

    def test_momentum_blocked_in_pre_open_and_closed(self) -> None:
        """Outside the trading session there is nothing to trade at all."""
        assert is_momentum_blocked(MarketPhase.PRE_OPEN) is False
        assert is_momentum_blocked(MarketPhase.CLOSED) is False


class TestLookupPerformance:
    def test_lookup_is_deterministic(self) -> None:
        """Same timestamp always yields the same phase — no clock state leaks."""
        ts = _ist_epoch(12, 0)
        first = get_market_phase(ts)
        for _ in range(1000):
            assert get_market_phase(ts) is first
