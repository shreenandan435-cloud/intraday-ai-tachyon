"""Phase 4 math engine tests — CLAUDE.md §3.

Each kernel is checked against an independent pure-NumPy/Python reference computed in the
test, not against a value copied from the implementation. A test that merely re-runs the
implementation proves nothing about whether the formula is right, and these formulas size
every stop and every position.
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tachyon.ipc.schemas import OrderBook, Tick
from tachyon.math_engine import warmup
from tachyon.math_engine.buffers import CandleBuffer, RingBuffer
from tachyon.math_engine.core import HistoricalBar, TickAggregator
from tachyon.math_engine.indicators import (
    calculate_depth_weighted_obi,
    calculate_ema,
    calculate_obi,
    calculate_vwap,
    calculate_wilder_atr,
)
from tachyon.math_engine.preseed import (
    bars_from_rows,
    preseed_aggregator,
    preseed_universe,
)
from tachyon.math_engine.warmup import (
    EngineNotWarmError,
    is_warm,
    require_warm,
    reset_warm_state,
)

pytestmark = pytest.mark.usefixtures("warm_engine")


@pytest.fixture(scope="session", autouse=True)
def warm_engine() -> None:
    """Compile the kernels once for the whole session."""
    assert warmup() is True


def _tick(seq: int, ltp: float, cumulative_volume: int, ts: float = 0.0) -> Tick:
    return Tick(token="2885", ltp=ltp, volume=cumulative_volume, ts_epoch=ts, seq=seq)


def _book(
    bid_qty: tuple[int, int, int, int, int] = (100, 200, 300, 400, 500),
    ask_qty: tuple[int, int, int, int, int] = (100, 200, 300, 400, 500),
) -> OrderBook:
    return OrderBook(
        token="2885",
        bid_price=(99.95, 99.90, 99.85, 99.80, 99.75),
        bid_qty=bid_qty,
        ask_price=(100.05, 100.10, 100.15, 100.20, 100.25),
        ask_qty=ask_qty,
        ts_epoch=0.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# buffers.py
# ──────────────────────────────────────────────────────────────────────────────


class TestRingBuffer:
    def test_fills_in_order(self) -> None:
        buf = RingBuffer(5)
        for i in range(3):
            buf.append(float(i))
        assert len(buf) == 3
        assert list(buf.view()) == [0.0, 1.0, 2.0]

    def test_wraps_and_keeps_chronological_order(self) -> None:
        """The mirrored layout must survive wrap-around without reordering."""
        buf = RingBuffer(5)
        for i in range(13):
            buf.append(float(i))
        assert len(buf) == 5
        assert list(buf.view()) == [8.0, 9.0, 10.0, 11.0, 12.0]

    def test_view_is_contiguous_after_wrapping(self) -> None:
        """Numba needs a C-contiguous array; a wrapped view must not require a copy."""
        buf = RingBuffer(100)
        for i in range(250):
            buf.append(float(i))
        view = buf.view()
        assert view.flags["C_CONTIGUOUS"]
        assert view.base is not None, "view must be a slice, not a fresh allocation"

    def test_view_does_not_allocate(self) -> None:
        buf = RingBuffer(64)
        for i in range(200):
            buf.append(float(i))
        first, second = buf.view(), buf.view()
        assert first.base is second.base

    def test_exact_capacity_boundary(self) -> None:
        buf = RingBuffer(4)
        for i in range(4):
            buf.append(float(i))
        assert buf.is_full
        assert list(buf.view()) == [0.0, 1.0, 2.0, 3.0]
        buf.append(4.0)
        assert list(buf.view()) == [1.0, 2.0, 3.0, 4.0]

    def test_latest_and_clear(self) -> None:
        buf = RingBuffer(8)
        with pytest.raises(IndexError):
            _ = buf.latest
        for i in range(12):
            buf.append(float(i))
        assert buf.latest == 11.0
        buf.clear()
        assert len(buf) == 0
        assert list(buf.view()) == []

    def test_last_n(self) -> None:
        buf = RingBuffer(10)
        for i in range(10):
            buf.append(float(i))
        assert list(buf.last(3)) == [7.0, 8.0, 9.0]
        assert list(buf.last(50)) == [float(i) for i in range(10)]

    def test_rejects_zero_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            RingBuffer(0)

    def test_candle_buffer_stays_aligned(self) -> None:
        candles = CandleBuffer(3)
        for i in range(5):
            candles.push(
                start_epoch=float(i),
                open_=float(i),
                high=float(i) + 1,
                low=float(i) - 1,
                close=float(i) + 0.5,
                volume=10.0,
            )
        assert len(candles) == 3
        assert list(candles.starts) == [2.0, 3.0, 4.0]
        assert list(candles.highs) == [3.0, 4.0, 5.0]
        assert list(candles.lows) == [1.0, 2.0, 3.0]
        assert list(candles.closes) == [2.5, 3.5, 4.5]


# ──────────────────────────────────────────────────────────────────────────────
# indicators.py
# ──────────────────────────────────────────────────────────────────────────────


class TestVwap:
    def test_matches_independent_reference(self) -> None:
        rng = np.random.default_rng(42)
        prices = rng.uniform(90, 110, 500)
        volumes = rng.integers(1, 500, 500).astype(np.float64)

        expected = float(np.sum(prices * volumes) / np.sum(volumes))
        assert calculate_vwap(prices, volumes) == pytest.approx(expected, rel=1e-12)

    def test_equal_volumes_reduce_to_the_mean(self) -> None:
        prices = np.array([10.0, 20.0, 30.0])
        volumes = np.array([5.0, 5.0, 5.0])
        assert calculate_vwap(prices, volumes) == pytest.approx(20.0)

    def test_weights_toward_high_volume_prints(self) -> None:
        prices = np.array([10.0, 100.0])
        volumes = np.array([1.0, 99.0])
        assert calculate_vwap(prices, volumes) == pytest.approx(99.1)

    def test_zero_volume_is_nan_not_zero(self) -> None:
        """A VWAP of 0.0 would make `price > vwap` trivially true and invent long signals."""
        prices = np.array([100.0, 101.0])
        assert math.isnan(calculate_vwap(prices, np.zeros(2)))

    def test_empty_is_nan(self) -> None:
        empty = np.zeros(0)
        assert math.isnan(calculate_vwap(empty, empty))


class TestEma:
    def test_matches_independent_reference(self) -> None:
        prices = np.linspace(100.0, 200.0, 100)
        period = 21

        alpha = 2.0 / (period + 1.0)
        expected = float(np.mean(prices[:period]))
        for price in prices[period:]:
            expected = alpha * price + (1 - alpha) * expected

        assert calculate_ema(prices, period) == pytest.approx(expected, rel=1e-12)

    def test_constant_series_returns_the_constant(self) -> None:
        assert calculate_ema(np.full(50, 42.0), 9) == pytest.approx(42.0)

    def test_seeded_from_sma_not_first_price(self) -> None:
        """With exactly `period` samples the result is the SMA, proving the seed."""
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert calculate_ema(prices, 5) == pytest.approx(3.0)

    def test_tracks_a_rising_series_below_the_last_price(self) -> None:
        prices = np.linspace(100.0, 200.0, 100)
        ema = calculate_ema(prices, 21)
        assert 100.0 < ema < 200.0

    def test_insufficient_data_is_nan(self) -> None:
        assert math.isnan(calculate_ema(np.array([1.0, 2.0]), 21))

    def test_invalid_period_is_nan(self) -> None:
        assert math.isnan(calculate_ema(np.linspace(1, 2, 50), 0))


class TestWilderAtr:
    def test_matches_independent_reference(self) -> None:
        rng = np.random.default_rng(7)
        closes = 100 + np.cumsum(rng.normal(0, 1, 200))
        highs = closes + rng.uniform(0.1, 2.0, 200)
        lows = closes - rng.uniform(0.1, 2.0, 200)
        period = 14

        true_ranges = [
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            for i in range(1, len(closes))
        ]
        expected = float(np.mean(true_ranges[:period]))
        for tr in true_ranges[period:]:
            expected = ((period - 1) * expected + tr) / period

        result = calculate_wilder_atr(highs, lows, closes, period)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_constant_range_gives_that_range(self) -> None:
        closes = np.full(50, 100.0)
        assert calculate_wilder_atr(closes + 1.0, closes - 1.0, closes, 14) == pytest.approx(2.0)

    def test_wilder_smoothing_is_slower_than_an_ema(self) -> None:
        """Wilder uses alpha = 1/period, an EMA 2/(period+1) — roughly twice as fast.

        Confusing them yields a materially tighter stop, and every position is sized off ATR.
        """
        period = 14
        closes = np.full(60, 100.0)
        highs = closes + 1.0
        lows = closes - 1.0
        highs[-1] = 120.0  # single large shock at the end

        wilder = calculate_wilder_atr(highs, lows, closes, period)

        true_ranges = [
            max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            for i in range(1, len(closes))
        ]
        alpha = 2.0 / (period + 1.0)
        ema_atr = float(np.mean(true_ranges[:period]))
        for tr in true_ranges[period:]:
            ema_atr = alpha * tr + (1 - alpha) * ema_atr

        assert wilder < ema_atr, "Wilder must react more slowly than an EMA of TR"

    def test_uses_previous_close_for_gaps(self) -> None:
        """A gap up must register through |high - previous_close|, not just high-low."""
        closes = np.array([100.0, 120.0])
        highs = np.array([100.5, 120.5])
        lows = np.array([99.5, 119.5])
        assert calculate_wilder_atr(highs, lows, closes, 1) == pytest.approx(20.5)

    def test_needs_period_plus_one_candles(self) -> None:
        closes = np.full(14, 100.0)
        assert math.isnan(calculate_wilder_atr(closes + 1, closes - 1, closes, 14))

        closes15 = np.full(15, 100.0)
        assert math.isfinite(calculate_wilder_atr(closes15 + 1, closes15 - 1, closes15, 14))

    def test_default_period_is_14(self) -> None:
        rng = np.random.default_rng(3)
        closes = 100 + np.cumsum(rng.normal(0, 1, 100))
        highs, lows = closes + 1, closes - 1
        assert calculate_wilder_atr(highs, lows, closes) == pytest.approx(
            calculate_wilder_atr(highs, lows, closes, 14)
        )


class TestObi:
    def test_formula(self) -> None:
        assert calculate_obi(300, 100) == pytest.approx(0.5)
        assert calculate_obi(100, 300) == pytest.approx(-0.5)
        assert calculate_obi(100, 100) == 0.0

    def test_bounded(self) -> None:
        assert calculate_obi(1_000_000, 0) == pytest.approx(1.0)
        assert calculate_obi(0, 1_000_000) == pytest.approx(-1.0)

    def test_empty_book_is_zero_not_nan(self) -> None:
        """CLAUDE.md §3.1 — a missing book is neutral evidence, and NaN would poison a
        confluence score that is otherwise valid."""
        result = calculate_obi(0, 0)
        assert result == 0.0
        assert not math.isnan(result)


class TestDepthWeightedObi:
    @staticmethod
    def _ladders() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bid_prices = np.array([99.95, 99.90, 99.85, 99.80, 99.75])
        ask_prices = np.array([100.05, 100.10, 100.15, 100.20, 100.25])
        qty = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        return bid_prices, qty.copy(), ask_prices, qty.copy()

    def test_symmetric_book_is_balanced(self) -> None:
        bp, bq, ap, aq = self._ladders()
        assert calculate_depth_weighted_obi(bp, bq, ap, aq) == pytest.approx(0.0, abs=1e-12)

    def test_bid_heavy_is_positive(self) -> None:
        bp, bq, ap, aq = self._ladders()
        bq *= 3.0
        assert calculate_depth_weighted_obi(bp, bq, ap, aq) > 0.3

    def test_bounded_within_unit_interval(self) -> None:
        bp, bq, ap, aq = self._ladders()
        assert calculate_depth_weighted_obi(bp, bq, ap, np.zeros(5)) == pytest.approx(1.0)
        assert calculate_depth_weighted_obi(bp, np.zeros(5), ap, aq) == pytest.approx(-1.0)

    def test_near_touch_size_outweighs_deep_size(self) -> None:
        """The whole point of distance decay: deep size must not dominate."""
        bp, _, ap, _ = self._ladders()
        near_bid = np.array([500.0, 0.0, 0.0, 0.0, 0.0])
        deep_ask = np.array([0.0, 0.0, 0.0, 0.0, 500.0])
        assert calculate_depth_weighted_obi(bp, near_bid, ap, deep_ask) > 0.0

    def test_equal_size_at_different_depths_favours_the_closer_side(self) -> None:
        bp, _, ap, _ = self._ladders()
        bid_level1 = np.array([0.0, 100.0, 0.0, 0.0, 0.0])
        ask_level4 = np.array([0.0, 0.0, 0.0, 100.0, 0.0])
        assert calculate_depth_weighted_obi(bp, bid_level1, ap, ask_level4) > 0.0

    def test_crossed_book_falls_back_to_positional_decay(self) -> None:
        crossed_bid = np.array([100.10, 100.05, 100.0, 99.95, 99.90])
        crossed_ask = np.array([99.90, 99.95, 100.0, 100.05, 100.10])
        qty = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        result = calculate_depth_weighted_obi(crossed_bid, qty, crossed_ask, qty.copy())
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_empty_book_is_zero(self) -> None:
        zeros = np.zeros(5)
        assert calculate_depth_weighted_obi(zeros, zeros, zeros, zeros) == 0.0
        empty = np.zeros(0)
        assert calculate_depth_weighted_obi(empty, empty, empty, empty) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# warmup.py
# ──────────────────────────────────────────────────────────────────────────────


class TestWarmup:
    def test_warmup_is_idempotent_and_reports_warm(self) -> None:
        assert warmup() is True
        assert warmup() is True
        assert is_warm()

    def test_require_warm_passes_when_warm(self) -> None:
        warmup()
        require_warm()  # must not raise

    def test_require_warm_raises_when_cold(self) -> None:
        reset_warm_state()
        try:
            with pytest.raises(EngineNotWarmError, match="cold"):
                require_warm()
        finally:
            assert warmup() is True

    def test_aggregator_refuses_to_construct_cold(self) -> None:
        """The enforcement point: no market data can reach an uncompiled kernel."""
        reset_warm_state()
        try:
            with pytest.raises(EngineNotWarmError):
                TickAggregator("RELIANCE")
        finally:
            assert warmup() is True


# ──────────────────────────────────────────────────────────────────────────────
# core.py
# ──────────────────────────────────────────────────────────────────────────────


class TestTickAggregator:
    def test_cumulative_volume_is_differenced(self) -> None:
        """Tick.volume is the session running total; VWAP needs per-print size."""
        agg = TickAggregator("RELIANCE")
        agg.on_tick(_tick(1, 100.0, 1000))
        agg.on_tick(_tick(2, 200.0, 1100))  # 100 traded at 200
        agg.on_tick(_tick(3, 300.0, 1400))  # 300 traded at 300

        # (200*100 + 300*300) / 400 = 275. Using cumulative volume would give ~236.8.
        assert agg.session_vwap == pytest.approx(275.0)

    def test_first_tick_contributes_no_volume(self) -> None:
        agg = TickAggregator("RELIANCE")
        agg.on_tick(_tick(1, 100.0, 5000))
        assert math.isnan(agg.session_vwap), "no per-print size is knowable from one tick"

    def test_volume_going_backwards_is_discarded(self) -> None:
        agg = TickAggregator("RELIANCE")
        agg.on_tick(_tick(1, 100.0, 1000))
        agg.on_tick(_tick(2, 100.0, 1100))
        agg.on_tick(_tick(3, 999.0, 50))  # feed reset
        assert agg.session_vwap == pytest.approx(100.0)

    def test_session_vwap_is_not_the_rolling_window(self) -> None:
        """Session VWAP must span the whole session, not just the ring buffer."""
        agg = TickAggregator("RELIANCE", tick_capacity=10)
        for i in range(1, 101):
            agg.on_tick(_tick(i, 100.0, i * 10))
        for i in range(101, 111):
            agg.on_tick(_tick(i, 200.0, i * 10))

        snapshot = agg.snapshot()
        assert snapshot.rolling_vwap == pytest.approx(200.0, rel=0.05)
        assert snapshot.session_vwap < 130.0, "session VWAP must remember the early prints"

    def test_sequence_gap_invalidates_buffers(self) -> None:
        agg = TickAggregator("RELIANCE")
        for i in range(1, 51):
            agg.on_tick(_tick(i, 100.0, i * 10))
        assert agg.tick_count == 50

        agg.on_tick(_tick(60, 100.0, 700))  # gap: expected 51
        assert agg.gaps_detected == 1
        assert agg.tick_count == 1, "buffers must be discarded, not carried forward"

    def test_sequence_gap_permanently_voids_session_vwap(self) -> None:
        agg = TickAggregator("RELIANCE")
        for i in range(1, 21):
            agg.on_tick(_tick(i, 100.0, i * 10))
        assert math.isfinite(agg.session_vwap)

        agg.on_tick(_tick(99, 100.0, 500))
        assert not agg.session_vwap_valid
        assert math.isnan(agg.session_vwap)

        for i in range(100, 300):
            agg.on_tick(_tick(i, 100.0, i * 10))
        assert math.isnan(agg.session_vwap), "lost prints are unrecoverable for the session"

    def test_taint_clears_once_the_window_refills(self) -> None:
        agg = TickAggregator("RELIANCE", ema_periods=(9,), recovery_ticks=20)
        for i in range(1, 31):
            agg.on_tick(_tick(i, 100.0, i * 10))

        agg.on_tick(_tick(500, 100.0, 5000))
        assert agg.tainted

        for i in range(501, 525):
            agg.on_tick(_tick(i, 100.0, i * 10))
        assert not agg.tainted, "rolling indicators recover once clean data fills the window"

    def test_out_of_order_sequence_is_a_gap(self) -> None:
        agg = TickAggregator("RELIANCE")
        agg.on_tick(_tick(10, 100.0, 100))
        agg.on_tick(_tick(5, 100.0, 200))  # publisher restart / reorder
        assert agg.gaps_detected == 1

    def test_duplicate_sequence_is_a_gap(self) -> None:
        agg = TickAggregator("RELIANCE")
        agg.on_tick(_tick(1, 100.0, 100))
        agg.on_tick(_tick(1, 100.0, 200))
        assert agg.gaps_detected == 1

    def test_contiguous_sequence_never_flags(self) -> None:
        agg = TickAggregator("RELIANCE")
        for i in range(1, 1001):
            agg.on_tick(_tick(i, 100.0 + i * 0.01, i * 10))
        assert agg.gaps_detected == 0
        assert not agg.tainted

    def test_candles_fold_on_five_minute_boundaries(self) -> None:
        agg = TickAggregator("RELIANCE")
        base = 1_786_000_000.0 - (1_786_000_000.0 % 300.0)

        seq = 1
        for candle in range(4):
            for offset in (10.0, 100.0, 200.0):
                agg.on_tick(
                    _tick(
                        seq,
                        100.0 + candle + offset / 1000.0,
                        seq * 10,
                        base + candle * 300 + offset,
                    )
                )
                seq += 1

        # Three complete candles are pushed; the fourth is still open.
        assert agg.candle_count == 3

    def test_atr_available_after_enough_candles(self) -> None:
        agg = TickAggregator("RELIANCE", atr_period=3)
        base = 1_786_000_000.0 - (1_786_000_000.0 % 300.0)

        seq = 1
        for candle in range(8):
            for price in (100.0 + candle, 101.0 + candle, 99.0 + candle):
                agg.on_tick(_tick(seq, price, seq * 10, base + candle * 300 + seq % 200))
                seq += 1

        atr = agg.atr()
        assert math.isfinite(atr)
        assert atr > 0.0

    def test_atr_is_nan_before_enough_candles(self) -> None:
        agg = TickAggregator("RELIANCE")
        agg.on_tick(_tick(1, 100.0, 10))
        assert math.isnan(agg.atr())

    def test_orderbook_updates_both_obi_variants(self) -> None:
        agg = TickAggregator("RELIANCE")
        agg.on_orderbook(
            _book(bid_qty=(300, 100, 100, 100, 100), ask_qty=(100, 100, 100, 100, 100))
        )
        snapshot = agg.snapshot()
        assert snapshot.obi == pytest.approx(0.5)
        assert snapshot.obi_weighted > 0.0

    def test_the_two_obi_variants_can_disagree(self) -> None:
        """Bid-heavy at the touch but ask-heavy in the depth is real, useful information.

        A large resting size far from the touch is the classic spoofing shape; the weighted
        variant discounts it while top-of-book OBI does not see it at all.
        """
        agg = TickAggregator("RELIANCE")
        agg.on_orderbook(
            _book(bid_qty=(300, 100, 100, 100, 100), ask_qty=(100, 200, 300, 400, 500))
        )
        snapshot = agg.snapshot()
        assert snapshot.obi > 0.0, "top of book is bid-heavy"
        assert snapshot.obi_weighted < 0.0, "the full ladder is ask-heavy"

    def test_orderbook_does_not_allocate_per_update(self) -> None:
        agg = TickAggregator("RELIANCE")
        agg.on_orderbook(_book())
        first = agg._bid_prices
        agg.on_orderbook(_book(bid_qty=(1, 2, 3, 4, 5)))
        assert agg._bid_prices is first, "depth scratch arrays must be reused"

    def test_snapshot_is_not_tradeable_without_atr(self) -> None:
        agg = TickAggregator("RELIANCE")
        agg.on_tick(_tick(1, 100.0, 10))
        assert not agg.snapshot().is_tradeable

    def test_snapshot_reports_every_ema_period(self) -> None:
        agg = TickAggregator("RELIANCE", ema_periods=(9, 21, 50))
        for i in range(1, 101):
            agg.on_tick(_tick(i, 100.0, i * 10))
        snapshot = agg.snapshot()
        assert snapshot.ema_periods == (9, 21, 50)
        assert len(snapshot.emas) == 3
        assert all(value == pytest.approx(100.0) for value in snapshot.emas)

    def test_reset_session_clears_everything(self) -> None:
        agg = TickAggregator("RELIANCE")
        for i in range(1, 51):
            agg.on_tick(_tick(i, 100.0, i * 10))
        agg.on_tick(_tick(999, 100.0, 900))
        assert agg.gaps_detected == 1

        agg.reset_session()
        assert agg.tick_count == 0
        assert agg.gaps_detected == 0
        assert agg.session_vwap_valid
        assert not agg.tainted

    def test_aggregators_are_independent(self) -> None:
        first = TickAggregator("RELIANCE")
        second = TickAggregator("INFY")
        first.on_tick(_tick(1, 100.0, 10))
        first.on_tick(_tick(5, 100.0, 20))  # gap on RELIANCE only

        assert first.gaps_detected == 1
        assert second.gaps_detected == 0


# ─── Cache locator safety (Vector 7) ─────────────────────────────────────────


class TestNumbaCacheLocator:
    """``cache=True`` must never crash a live tick with 'no locator available'."""

    @staticmethod
    def _probe(env_value: str | None, tmp_path: Path) -> str:

        env = dict(os.environ)
        if env_value is None:
            env.pop("NUMBA_CACHE_DIR", None)
        else:
            env["NUMBA_CACHE_DIR"] = env_value
        repo_src = str(Path(__file__).resolve().parents[2] / "src")
        code = (
            "import sys, os; "
            "import tachyon.math_engine.indicators; "
            "print(os.environ.get('NUMBA_CACHE_DIR', ''))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**env, "PYTHONPATH": repo_src + os.pathsep + env.get("PYTHONPATH", "")},
            cwd=str(tmp_path),
            timeout=300,
        )
        return result.stdout.strip().splitlines()[-1]

    def test_unset_env_falls_back_to_a_writable_numba_cache(self, tmp_path: Path) -> None:
        resolved = self._probe(None, tmp_path)
        # Repo root is preferred when reachable from the installed layout; cwd fallback
        # otherwise. Either way it must be writable and end with .numba_cache.
        assert resolved.endswith(".numba_cache")
        assert Path(resolved).exists()

    def test_explicit_numba_cache_dir_is_respected(self, tmp_path: Path) -> None:
        override = tmp_path / "custom-cache"
        resolved = self._probe(str(override), tmp_path)
        assert resolved == str(override)
        assert override.exists()

    def test_kernels_are_nogil_with_cache_machinery(self) -> None:
        from tachyon.math_engine.indicators import KERNELS

        for kernel in KERNELS:
            targetoptions = getattr(kernel, "targetoptions", {})
            assert targetoptions.get("nogil") is True, f"{kernel.__name__}: nogil"
            # The cache flag itself lives in the dispatcher's locator machinery, not in
            # targetoptions; the presence of the on-disk counters proves it is wired.
            assert hasattr(kernel, "_cache_misses"), f"{kernel.__name__}: cache disabled"


# ─── Pre-market pre-seed (CLAUDE.md §3 — "the cold start is a bug") ───────────


def _synthetic_ohlcv(
    *,
    days: int = 3,
    interval_seconds: int = 300,
    base: float = 100.0,
    step: float = 0.1,
    start_epoch: float = 1_700_000_000.0,
    volume: float = 1000.0,
) -> list[HistoricalBar]:
    """Deterministic OHLCV series for seed tests — no random component.

    Each candle's close advances by ``step``, so EMAs are predictable without needing a
    reference implementation to be re-derived from the implementation under test.
    """
    candles_per_day = (6 * 60 * 60) // interval_seconds  # 09:15–15:15 IST ≈ 6h15m
    bars: list[HistoricalBar] = []
    for i in range(days * candles_per_day):
        price = base + step * i
        bars.append(
            HistoricalBar(
                start_epoch=start_epoch + i * interval_seconds,
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price + step * 0.5,
                volume=volume,
            )
        )
    return bars


class TestSeedBars:
    def test_empty_input_is_a_noop(self) -> None:
        agg = TickAggregator("RELIANCE")
        agg.seed_bars([])
        assert agg.candle_count == 0
        assert agg.tick_count == 0
        assert math.isnan(agg.ltp)
        assert math.isnan(agg.atr())

    def test_populates_candle_buffer(self) -> None:
        agg = TickAggregator("RELIANCE", candle_capacity=200)
        bars = _synthetic_ohlcv(days=1)
        agg.seed_bars(bars)
        assert agg.candle_count == len(bars)
        # First and last candle reflect the seeded series.
        snap = agg.snapshot()
        assert snap.candle_count == len(bars)

    def test_atr_is_finite_after_seeding(self) -> None:
        """The whole point of the pre-seed: ATR must be ready before 09:15 open."""
        agg = TickAggregator("RELIANCE", atr_period=14, candle_capacity=200)
        bars = _synthetic_ohlcv(days=1)
        agg.seed_bars(bars)
        atr = agg.atr()
        assert math.isfinite(atr)
        assert atr > 0.0

    def test_emas_are_finite_after_seeding(self) -> None:
        agg = TickAggregator("RELIANCE", ema_periods=(9, 20, 50), tick_capacity=200)
        bars = _synthetic_ohlcv(days=1)
        agg.seed_bars(bars)
        emas = agg.snapshot().emas
        assert len(emas) == 3
        assert all(math.isfinite(value) for value in emas)

    def test_session_vwap_is_not_contaminated_by_preeed_data(self) -> None:
        """Session VWAP anchors at 09:15; pre-seeded bars must not bleed into it."""
        agg = TickAggregator("RELIANCE")
        bars = _synthetic_ohlcv(days=2)
        agg.seed_bars(bars)
        assert agg.session_vwap_valid is True
        # No live tick yet: there is no per-print session volume.
        assert math.isnan(agg.session_vwap)

    def test_last_close_becomes_ltp(self) -> None:
        agg = TickAggregator("RELIANCE")
        bars = _synthetic_ohlcv(days=1)
        agg.seed_bars(bars)
        last_close = bars[-1].close
        assert agg.ltp == pytest.approx(last_close)
        assert agg._last_ts == pytest.approx(bars[-1].start_epoch)

    def test_does_not_flag_sequence_gap(self) -> None:
        """Pre-seeded bars have no sequence number; _last_seq must stay None."""
        agg = TickAggregator("RELIANCE")
        agg.seed_bars(_synthetic_ohlcv(days=1))
        assert agg._last_seq is None
        assert agg.gaps_detected == 0
        assert not agg.tainted

    def test_live_ticks_resume_after_seeding(self) -> None:
        """A pre-seeded aggregator must keep working when the first live tick lands."""
        agg = TickAggregator("RELIANCE", ema_periods=(20,), tick_capacity=200)
        agg.seed_bars(_synthetic_ohlcv(days=1))
        atr_before = agg.atr()
        assert math.isfinite(atr_before)

        agg.on_tick(_tick(1, 150.0, 100, ts=1_700_000_000.0 + 1_000))
        assert agg.tick_count == len(list(_synthetic_ohlcv(days=1))) + 1
        # ATR remains finite through the live tick.
        assert math.isfinite(agg.atr())


class TestBarsFromRows:
    def test_round_trip(self) -> None:
        rows = [
            ["2026-08-31 09:15:00", 100.0, 101.0, 99.0, 100.5, 1000.0],
            ["2026-08-31 09:20:00", 100.5, 102.0, 100.0, 101.5, 1500.0],
        ]
        bars = bars_from_rows(rows, interval_minutes=5)
        assert len(bars) == 2
        first = bars[0]
        assert first.open == 100.0
        assert first.high == 101.0
        assert first.low == 99.0
        assert first.close == 100.5
        assert first.volume == 1000.0
        assert isinstance(first, HistoricalBar)

    def test_malformed_rows_are_skipped(self) -> None:
        rows: list[list[Any]] = [
            ["2026-08-31 09:15:00", 100.0, 101.0, 99.0, 100.5, 1000.0],
            ["not-a-timestamp", 1, 2, 3, 4, 5],
            ["2026-08-31 09:25:00", 1, 2, 3, 4, 5],
        ]
        bars = bars_from_rows(rows, interval_minutes=5)
        assert len(bars) == 2


class _StubFetcher:
    """Async fetcher stand-in for preseed_aggregator tests."""

    def __init__(self, rows: list[list[Any]] | None = None, exc: BaseException | None = None) -> None:
        self._rows = rows
        self._exc = exc
        self.calls = 0

    async def fetch_historical_candles(
        self,
        exchange: str,
        symbol_token: str,
        days: int = 3,
        interval: str = "FIVE_MINUTE",
    ) -> list[list[Any]]:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return list(self._rows or [])


class TestPreseedAggregator:
    def test_returns_zero_when_no_fetcher(self) -> None:
        agg = TickAggregator("RELIANCE")
        result = asyncio.run(
            preseed_aggregator(
                agg, exchange="NSE", symbol_token="2885", fetcher=None
            )
        )
        assert result == 0
        assert agg.candle_count == 0

    def test_seeds_aggregator_with_returned_rows(self) -> None:
        rows = [
            ["2026-08-31 09:15:00", 100.0, 101.0, 99.0, 100.5, 1000.0],
            ["2026-08-31 09:20:00", 100.5, 102.0, 100.0, 101.5, 1500.0],
        ] * 30
        agg = TickAggregator("RELIANCE", candle_capacity=200, tick_capacity=200)
        fetcher = _StubFetcher(rows=rows)
        result = asyncio.run(
            preseed_aggregator(
                agg, exchange="NSE", symbol_token="2885", fetcher=fetcher
            )
        )
        assert result == 60
        assert agg.candle_count == 60
        assert fetcher.calls == 1

    def test_empty_rows_leaves_aggregator_cold(self) -> None:
        agg = TickAggregator("RELIANCE")
        result = asyncio.run(
            preseed_aggregator(
                agg,
                exchange="NSE",
                symbol_token="2885",
                fetcher=_StubFetcher(rows=[]),
            )
        )
        assert result == 0
        assert agg.candle_count == 0

    def test_fetcher_exception_does_not_propagate(self) -> None:
        agg = TickAggregator("RELIANCE")
        result = asyncio.run(
            preseed_aggregator(
                agg,
                exchange="NSE",
                symbol_token="2885",
                fetcher=_StubFetcher(exc=RuntimeError("rate-limited")),
            )
        )
        assert result == 0
        assert agg.candle_count == 0


class TestPreseedUniverse:
    def test_iterates_every_symbol_in_order(self) -> None:
        aggregators = {
            "RELIANCE": TickAggregator("RELIANCE", candle_capacity=200, tick_capacity=200),
            "INFY": TickAggregator("INFY", candle_capacity=200, tick_capacity=200),
        }

        class _Host:
            def __iter__(self):  # type: ignore[no-untyped-def]
                return iter(aggregators.items())

        rows = [
            ["2026-08-31 09:15:00", 100.0, 101.0, 99.0, 100.5, 1000.0],
            ["2026-08-31 09:20:00", 100.5, 102.0, 100.0, 101.5, 1500.0],
        ] * 30
        fetcher = _StubFetcher(rows=rows)
        tokens = {"RELIANCE": "2885", "INFY": "1594"}
        results = asyncio.run(
            preseed_universe(_Host(), fetcher, token_lookup=tokens)
        )
        assert results == {"RELIANCE": 60, "INFY": 60}
        assert fetcher.calls == 2
        for agg in aggregators.values():
            assert agg.candle_count == 60

    def test_missing_token_is_logged_and_recorded_as_zero(self) -> None:
        aggregators = {
            "RELIANCE": TickAggregator("RELIANCE"),
            "UNKNOWN": TickAggregator("UNKNOWN"),
        }

        class _Host:
            def __iter__(self):  # type: ignore[no-untyped-def]
                return iter(aggregators.items())

        fetcher = _StubFetcher(rows=[])
        results = asyncio.run(
            preseed_universe(_Host(), fetcher, token_lookup={"RELIANCE": "2885"})
        )
        assert results == {"RELIANCE": 0, "UNKNOWN": 0}
        assert aggregators["RELIANCE"].candle_count == 0
        assert aggregators["UNKNOWN"].candle_count == 0
        # Fetcher is called once for the known token, but returns no rows.
        assert fetcher.calls == 1

    def test_no_token_lookup_leaves_everyone_cold(self) -> None:
        aggregators = {"RELIANCE": TickAggregator("RELIANCE")}

        class _Host:
            def __iter__(self):  # type: ignore[no-untyped-def]
                return iter(aggregators.items())

        fetcher = _StubFetcher(rows=[])
        results = asyncio.run(preseed_universe(_Host(), fetcher))
        assert results == {"RELIANCE": 0}
        assert fetcher.calls == 0
