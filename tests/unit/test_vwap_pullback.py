"""Tests for the VWAP-band pullback strategy."""

from __future__ import annotations

import math

import pytest

from tachyon.math_engine.buffers import CandleBuffer
from tachyon.strategies.time_gates import MarketPhase
from tachyon.strategies.vwap_pullback import (
    BAND_WIDTH_SIGMA,
    MIN_CANDLES_FOR_BANDS,
    PullbackSide,
    PullbackSignal,
    VwapPullbackStrategy,
    WICK_TO_BODY_RATIO,
    candle_pattern,
    compute_vwap_bands,
)


def _push(candles: CandleBuffer, *, open_: float, high: float, low: float, close: float, volume: float = 1000.0, start: float | None = None) -> None:
    idx = len(candles)
    candles.push(
        start_epoch=start if start is not None else float(idx * 300),
        open_=open_, high=high, low=low, close=close, volume=volume,
    )


class TestComputeVwapBands:
    def test_empty_buffer_is_invalid(self) -> None:
        bands = compute_vwap_bands(CandleBuffer(10))
        assert not bands.is_valid
        assert bands.sample_count == 0

    def test_too_few_candles_is_invalid(self) -> None:
        candles = CandleBuffer(10)
        for i in range(MIN_CANDLES_FOR_BANDS - 1):
            _push(candles, open_=100.0, high=100.5, low=99.5, close=100.2, volume=1000.0)
        assert not compute_vwap_bands(candles).is_valid

    def test_flat_series_has_zero_std_dev(self) -> None:
        candles = CandleBuffer(10)
        for _ in range(MIN_CANDLES_FOR_BANDS + 2):
            _push(candles, open_=100.0, high=100.0, low=100.0, close=100.0, volume=1000.0)
        bands = compute_vwap_bands(candles)
        # Flat → σ = 0 → invalid (no useful band).
        assert not bands.is_valid
        assert bands.std_dev == 0.0

    def test_valid_bands_bracket_vwap(self) -> None:
        candles = CandleBuffer(20)
        for i in range(MIN_CANDLES_FOR_BANDS + 2):
            p = 100.0 + float(i) * 0.1
            _push(candles, open_=p, high=p + 0.5, low=p - 0.5, close=p + 0.05, volume=1000.0)
        bands = compute_vwap_bands(candles)
        assert bands.is_valid
        assert bands.upper > bands.vwap > bands.lower
        assert math.isclose(bands.upper - bands.vwap, bands.vwap - bands.lower)

    def test_zero_total_volume_is_invalid(self) -> None:
        candles = CandleBuffer(10)
        for _ in range(MIN_CANDLES_FOR_BANDS + 1):
            _push(candles, open_=100.0, high=100.5, low=99.5, close=100.2, volume=0.0)
        assert not compute_vwap_bands(candles).is_valid


class TestCandlePattern:
    def test_hammer_is_bullish_reversal(self) -> None:
        candles = CandleBuffer(10)
        # Bullish body of 0.5; lower wick of 3.0 (6x body — well past 2:1).
        _push(candles, open_=100.0, high=100.5, low=97.0, close=100.5, volume=1000.0)
        pattern = candle_pattern(candles)
        assert pattern.is_bullish_reversal is True
        assert pattern.is_bearish_reversal is False

    def test_shooting_star_is_bearish_reversal(self) -> None:
        candles = CandleBuffer(10)
        # Bearish body of 0.5; upper wick of 3.0.
        _push(candles, open_=100.0, high=103.0, low=99.5, close=99.5, volume=1000.0)
        pattern = candle_pattern(candles)
        assert pattern.is_bearish_reversal is True
        assert pattern.is_bullish_reversal is False

    def test_wick_below_ratio_is_not_reversal(self) -> None:
        candles = CandleBuffer(10)
        # Lower wick 0.9x body — below the 2:1 threshold.
        _push(candles, open_=100.0, high=100.5, low=99.55, close=100.5, volume=1000.0)
        pattern = candle_pattern(candles)
        assert pattern.is_bullish_reversal is False

    def test_doji_is_not_reversal(self) -> None:
        candles = CandleBuffer(10)
        _push(candles, open_=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)
        pattern = candle_pattern(candles)
        assert pattern.is_bullish_reversal is False
        assert pattern.is_bearish_reversal is False

    def test_empty_buffer_is_not_reversal(self) -> None:
        pattern = candle_pattern(CandleBuffer(10))
        assert math.isnan(pattern.open)
        assert pattern.is_bullish_reversal is False
        assert pattern.is_bearish_reversal is False

    def test_wrong_direction_close_is_not_reversal(self) -> None:
        candles = CandleBuffer(10)
        # Lower wick long enough, but Close < Open → not bullish.
        _push(candles, open_=100.0, high=100.5, low=97.0, close=99.0, volume=1000.0)
        pattern = candle_pattern(candles)
        assert pattern.is_bullish_reversal is False
        assert pattern.is_bearish_reversal is False


class TestStrategyActivity:
    def test_exclusive_to_midday_chop(self) -> None:
        s = VwapPullbackStrategy("RELIANCE")
        assert s.is_active_in(MarketPhase.MIDDAY_CHOP) is True
        for phase in (
            MarketPhase.PRE_OPEN,
            MarketPhase.ORB_BUILD,
            MarketPhase.GOLDEN_WINDOW,
            MarketPhase.AFTERNOON_TREND,
            MarketPhase.SQUARE_OFF,
            MarketPhase.CLOSED,
        ):
            assert s.is_active_in(phase) is False


class TestEvaluate:
    def _make_candles(self) -> CandleBuffer:
        """Build 8 bars: first 7 range-bound, last one a hammer at the lower band."""
        candles = CandleBuffer(20)
        # Build bars with ascending then descending prices; last bar is a hammer.
        prices = [100, 101, 102, 103, 102, 101, 100]
        for p in prices:
            _push(candles, open_=float(p), high=p + 0.5, low=p - 0.5, close=p + 0.3, volume=1000.0)
        # Hammer at the low: open 99, low 95, close 99.5 → bullish body 0.5, lower wick 4.0
        _push(candles, open_=99.0, high=99.7, low=95.0, close=99.5, volume=1000.0)
        return candles

    def test_long_signal_in_midday_chop(self) -> None:
        candles = self._make_candles()
        s = VwapPullbackStrategy("RELIANCE")
        signal = s.evaluate(
            phase=MarketPhase.MIDDAY_CHOP,
            candles=candles,
            ltp=99.5, atr_5m=2.0,
            best_bid=99.0, best_ask=99.2,
            ts_epoch=0.0,
        )
        assert signal is not None
        assert signal.is_long
        assert signal.side is PullbackSide.LONG
        # Long price = best_ask * 1.0005
        assert float(signal.price) == pytest.approx(99.2 * 1.0005, rel=1e-9)
        assert signal.vwap > 0
        assert signal.std_dev > 0
        assert signal.band < signal.vwap  # lower band

    def test_no_signal_outside_midday_chop(self) -> None:
        candles = self._make_candles()
        s = VwapPullbackStrategy("RELIANCE")
        for phase in (
            MarketPhase.ORB_BUILD,
            MarketPhase.GOLDEN_WINDOW,
            MarketPhase.AFTERNOON_TREND,
            MarketPhase.SQUARE_OFF,
        ):
            assert s.evaluate(
                phase=phase, candles=candles, ltp=99.5, atr_5m=2.0,
                best_bid=99.0, best_ask=99.2, ts_epoch=0.0,
            ) is None

    def test_no_signal_when_atr_unavailable(self) -> None:
        candles = self._make_candles()
        s = VwapPullbackStrategy("RELIANCE")
        assert s.evaluate(
            phase=MarketPhase.MIDDAY_CHOP, candles=candles, ltp=99.5, atr_5m=math.nan,
            best_bid=99.0, best_ask=99.2, ts_epoch=0.0,
        ) is None
        assert s.evaluate(
            phase=MarketPhase.MIDDAY_CHOP, candles=candles, ltp=99.5, atr_5m=0.0,
            best_bid=99.0, best_ask=99.2, ts_epoch=0.0,
        ) is None

    def test_no_signal_with_crossed_book(self) -> None:
        candles = self._make_candles()
        s = VwapPullbackStrategy("RELIANCE")
        # Crossed book: best_bid > best_ask.
        assert s.evaluate(
            phase=MarketPhase.MIDDAY_CHOP, candles=candles, ltp=99.5, atr_5m=2.0,
            best_bid=100.0, best_ask=99.0, ts_epoch=0.0,
        ) is None

    def test_no_signal_when_reversal_pattern_missing(self) -> None:
        """Bar reaches the lower band but the candle is not a hammer → no signal."""
        candles = CandleBuffer(20)
        prices = [100, 101, 102, 103, 102, 101, 100]
        for p in prices:
            _push(candles, open_=float(p), high=p + 0.5, low=p - 0.5, close=p + 0.3, volume=1000.0)
        # Long-bodied bullish bar that prints a low below the lower band but
        # whose lower wick is not twice the body — not a reversal.
        _push(candles, open_=98.0, high=101.0, low=95.0, close=100.5, volume=1000.0)
        s = VwapPullbackStrategy("RELIANCE")
        assert s.evaluate(
            phase=MarketPhase.MIDDAY_CHOP, candles=candles, ltp=99.0, atr_5m=2.0,
            best_bid=99.0, best_ask=99.5, ts_epoch=0.0,
        ) is None

    def test_short_signal_in_midday_chop(self) -> None:
        candles = CandleBuffer(20)
        prices = [100, 101, 102, 103, 102, 101, 100]
        for p in prices:
            _push(candles, open_=float(p), high=p + 0.5, low=p - 0.5, close=p + 0.3, volume=1000.0)
        # Shooting star at the top: open 100, high 105, close 99.5 → bearish body 0.5, upper wick 5.0
        _push(candles, open_=100.0, high=105.0, low=99.5, close=99.5, volume=1000.0)
        s = VwapPullbackStrategy("RELIANCE")
        signal = s.evaluate(
            phase=MarketPhase.MIDDAY_CHOP, candles=candles, ltp=99.5, atr_5m=2.0,
            best_bid=99.0, best_ask=99.3, ts_epoch=0.0,
        )
        assert signal is not None
        assert signal.is_short
        # Short price = best_bid * 0.9995
        assert float(signal.price) == pytest.approx(99.0 * 0.9995, rel=1e-9)
        assert signal.band > signal.vwap  # upper band


class TestSignalIntegrity:
    def test_rejects_nan_ltp(self) -> None:
        with pytest.raises(ValueError, match="ltp"):
            PullbackSignal(
                symbol="RELIANCE",
                side=PullbackSide.LONG,
                price=__import__("decimal").Decimal("100"),
                vwap=100.0, band=99.0, band_distance=1.0,
                std_dev=1.0, ltp=math.nan, atr_5m=1.0, ts_epoch=0.0,
            )

    def test_rejects_zero_std_dev(self) -> None:
        with pytest.raises(ValueError, match="σ"):
            PullbackSignal(
                symbol="RELIANCE",
                side=PullbackSide.LONG,
                price=__import__("decimal").Decimal("100"),
                vwap=100.0, band=99.0, band_distance=1.0,
                std_dev=0.0, ltp=100.0, atr_5m=1.0, ts_epoch=0.0,
            )


class TestModuleConstants:
    def test_thresholds_match_spec(self) -> None:
        assert BAND_WIDTH_SIGMA == pytest.approx(1.0)
        assert WICK_TO_BODY_RATIO == pytest.approx(2.0)
        assert MIN_CANDLES_FOR_BANDS == 6
