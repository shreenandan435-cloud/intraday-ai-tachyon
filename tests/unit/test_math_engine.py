"""Phase 4 math engine tests — CLAUDE.md §3.

Each kernel is checked against an independent pure-NumPy/Python reference computed in the
test, not against a value copied from the implementation. A test that merely re-runs the
implementation proves nothing about whether the formula is right, and these formulas size
every stop and every position.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tachyon.ipc.schemas import OrderBook, Tick
from tachyon.math_engine import warmup
from tachyon.math_engine.buffers import CandleBuffer, RingBuffer
from tachyon.math_engine.core import TickAggregator
from tachyon.math_engine.indicators import (
    calculate_depth_weighted_obi,
    calculate_ema,
    calculate_obi,
    calculate_vwap,
    calculate_wilder_atr,
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
