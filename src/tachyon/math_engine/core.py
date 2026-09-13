"""Tick aggregation and indicator orchestration — CLAUDE.md §3.

:class:`TickAggregator` owns one symbol's ring buffers, folds ticks into 5-minute candles, and
drives the JIT kernels. One instance per symbol; instances share nothing.

Three correctness concerns dominate this module.

**Sequence gaps.** The transport drops messages by design when a consumer falls behind
(CLAUDE.md §2.1). A gap in ``Tick.seq`` means prints are permanently lost, so every
accumulator built from them is wrong. On detection the aggregator logs ``CRITICAL``, discards
its buffers, and rebuilds from the next clean tick.

**Cumulative volume.** ``Tick.volume`` is the exchange's running session total, not the size
of that print. Feeding it straight into a VWAP produces a number that looks plausible and is
badly wrong — later prints carry enormous weight. The aggregator differences it before
anything sees it.

**Session VWAP cannot recover from a gap.** Rolling indicators heal once the buffer refills
with clean data, but session VWAP is an accumulation over *every* print since 09:15. Prints
lost to a gap are unrecoverable, so :attr:`IndicatorSnapshot.session_vwap` returns ``NaN`` for
the remainder of the session rather than a plausible lie. This is deliberate: a subtly wrong
VWAP anchoring the day's entries is worse than no VWAP at all.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Self

import numpy as np
import numpy.typing as npt

from tachyon.core.config import Settings
from tachyon.core.constants import ATR_PERIOD, CANDLE_INTERVAL_MINUTES
from tachyon.core.logger import get_logger
from tachyon.ipc.schemas import DEPTH_LEVELS, OrderBook, Tick
from tachyon.math_engine.buffers import FLOAT, CandleBuffer, RingBuffer
from tachyon.math_engine.indicators import (
    calculate_depth_weighted_obi,
    calculate_emas,
    calculate_obi,
    calculate_vwap,
    calculate_wilder_atr,
)
from tachyon.math_engine.warmup import require_warm

_log = get_logger(__name__)

DEFAULT_EMA_PERIODS: Final[tuple[int, ...]] = (9, 21, 50)
DEFAULT_TICK_CAPACITY: Final[int] = 2000
DEFAULT_CANDLE_CAPACITY: Final[int] = 500
CANDLE_SECONDS: Final[float] = float(CANDLE_INTERVAL_MINUTES * 60)


@dataclass(frozen=True, slots=True)
class HistoricalBar:
    """One historical OHLCV candle, in the math engine's internal bar representation.

    Mirrors the row shape produced by
    :func:`tachyon.ingestion.angel_adapter.parse_historical_rows`: ``start_epoch`` is the
    candle's bucket start in epoch seconds (already snapped to the 5-minute grid), and the
    remaining fields are the standard open/high/low/close/volume tuple.
    """

    start_epoch: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Immutable read of every indicator at one instant.

    ``NaN`` in any field means *not ready* — never zero. Callers must check with
    :func:`math.isnan` and produce no signal rather than treating it as a value.
    """

    symbol: str
    ts_epoch: float
    ltp: float
    session_vwap: float
    rolling_vwap: float
    ema_periods: tuple[int, ...]
    emas: tuple[float, ...]
    atr_5m: float
    obi: float
    obi_weighted: float
    tick_count: int
    candle_count: int
    tainted: bool
    session_vwap_valid: bool
    gaps_detected: int

    @property
    def is_tradeable(self) -> bool:
        """True only when every value the order geometry depends on is present and clean.

        ATR is load-bearing: stop distance, both targets and position size all derive from it
        (CLAUDE.md §6.1). Without a finite ATR there is no legitimate order to place.
        """
        return (
            not self.tainted
            and math.isfinite(self.atr_5m)
            and self.atr_5m > 0.0
            and math.isfinite(self.ltp)
        )


class TickAggregator:
    """Per-symbol buffers, candle folding and indicator computation.

    Args:
        symbol: watchlist symbol this instance tracks.
        tick_capacity: ticks retained for rolling VWAP and the EMAs.
        candle_capacity: completed 5-minute candles retained for ATR.
        ema_periods: EMA lookbacks to compute on every snapshot.
        atr_period: Wilder period. Defaults to the frozen constant.
        recovery_ticks: clean ticks required after a gap before rolling indicators are
            trusted again. Defaults to the longest EMA period, since that is the point at
            which no pre-gap sample can still influence the result.

    Raises:
        EngineNotWarmError: if the JIT kernels have not been compiled. Constructing an
            aggregator is the last checkpoint before live data reaches a kernel, so the guard
            lives here (CLAUDE.md §3).
    """

    __slots__ = (
        "_ask_prices",
        "_ask_qtys",
        "_atr_period",
        "_bid_prices",
        "_bid_qtys",
        "_candle_close",
        "_candle_high",
        "_candle_low",
        "_candle_open",
        "_candle_start",
        "_candle_volume",
        "_candles",
        "_ema_out",
        "_ema_periods",
        "_ema_periods_arr",
        "_gaps",
        "_last_cum_volume",
        "_last_seq",
        "_last_ts",
        "_last_volume_delta",
        "_ltp",
        "_obi",
        "_obi_weighted",
        "_prices",
        "_pv_sum",
        "_recovery_ticks",
        "_session_vwap_valid",
        "_v_sum",
        "_volumes",
        "symbol",
    )

    def __init__(
        self,
        symbol: str,
        *,
        tick_capacity: int = DEFAULT_TICK_CAPACITY,
        candle_capacity: int = DEFAULT_CANDLE_CAPACITY,
        ema_periods: tuple[int, ...] = DEFAULT_EMA_PERIODS,
        atr_period: int = ATR_PERIOD,
        recovery_ticks: int | None = None,
    ) -> None:
        require_warm()

        self.symbol = symbol
        self._ema_periods = ema_periods
        self._atr_period = atr_period
        self._recovery_ticks = (
            recovery_ticks if recovery_ticks is not None else max(ema_periods, default=50)
        )

        # Fused multi-period EMA scratch, allocated once: the int64 period vector the
        # kernel receives and the float64 output it fills on every snapshot. Reusing both
        # keeps snapshot() free of per-call allocation (CLAUDE.md §3.2).
        self._ema_periods_arr: npt.NDArray[np.int64] = np.array(ema_periods, dtype=np.int64)
        self._ema_out: npt.NDArray[np.float64] = np.empty(len(ema_periods), dtype=np.float64)

        self._prices = RingBuffer(tick_capacity, FLOAT)
        self._volumes = RingBuffer(tick_capacity, FLOAT)
        self._candles = CandleBuffer(candle_capacity)

        # Scratch depth arrays, allocated once. Converting the OrderBook tuples to fresh
        # NumPy arrays on every depth update would allocate on the hot path.
        self._bid_prices: npt.NDArray[np.float64] = np.zeros(DEPTH_LEVELS, dtype=np.float64)
        self._bid_qtys: npt.NDArray[np.float64] = np.zeros(DEPTH_LEVELS, dtype=np.float64)
        self._ask_prices: npt.NDArray[np.float64] = np.zeros(DEPTH_LEVELS, dtype=np.float64)
        self._ask_qtys: npt.NDArray[np.float64] = np.zeros(DEPTH_LEVELS, dtype=np.float64)

        self._last_seq: int | None = None
        self._last_cum_volume: int | None = None
        self._last_volume_delta = 0.0
        self._gaps = 0
        self._session_vwap_valid = True

        self._pv_sum = 0.0
        self._v_sum = 0.0

        self._ltp = math.nan
        self._last_ts = 0.0
        self._obi = 0.0
        self._obi_weighted = 0.0

        self._candle_start: float | None = None
        self._candle_open = 0.0
        self._candle_high = 0.0
        self._candle_low = 0.0
        self._candle_close = 0.0
        self._candle_volume = 0.0

    @classmethod
    def from_settings(cls, symbol: str, settings: Settings) -> Self:
        """Build using the tunables in ``config/settings.yaml``."""
        engine = settings.math_engine
        return cls(
            symbol,
            tick_capacity=engine.tick_buffer_size,
            candle_capacity=engine.candle_buffer_size,
            ema_periods=engine.ema_periods,
        )

    # ── ingestion ────────────────────────────────────────────────────────────

    def on_tick(self, tick: Tick) -> None:
        """Fold one tick into the buffers, candles and session accumulators."""
        if self._last_seq is not None and tick.seq != self._last_seq + 1:
            self._handle_gap(tick.seq)
        self._last_seq = tick.seq

        volume_delta = self._volume_delta(tick.volume)
        self._last_volume_delta = volume_delta

        self._ltp = tick.ltp
        self._last_ts = tick.ts_epoch
        self._prices.append(tick.ltp)
        self._volumes.append(volume_delta)

        if self._session_vwap_valid:
            self._pv_sum += tick.ltp * volume_delta
            self._v_sum += volume_delta

        self._fold_candle(tick.ts_epoch, tick.ltp, volume_delta)

    def on_orderbook(self, book: OrderBook) -> None:
        """Recompute both OBI variants from a depth snapshot."""
        self._bid_prices[:] = book.bid_price
        self._bid_qtys[:] = book.bid_qty
        self._ask_prices[:] = book.ask_price
        self._ask_qtys[:] = book.ask_qty

        self._obi = float(calculate_obi(book.bid_qty[0], book.ask_qty[0]))
        self._obi_weighted = float(
            calculate_depth_weighted_obi(
                self._bid_prices, self._bid_qtys, self._ask_prices, self._ask_qtys
            )
        )

    def _volume_delta(self, cumulative: int) -> float:
        """Convert the exchange's cumulative session volume into this print's size.

        The first tick yields ``0.0``: we cannot know how much of the running total traded at
        this price. A *decrease* means the feed reset or reconnected mid-session, so the delta
        is discarded rather than allowed to go negative and corrupt the VWAP numerator.
        """
        previous = self._last_cum_volume
        self._last_cum_volume = cumulative
        if previous is None:
            return 0.0
        delta = cumulative - previous
        if delta < 0:
            _log.warning(
                "math_engine.volume_went_backwards",
                symbol=self.symbol,
                previous=previous,
                received=cumulative,
                action="delta discarded",
            )
            return 0.0
        return float(delta)

    def _handle_gap(self, received_seq: int) -> None:
        """Discard everything derived from a broken sequence."""
        expected = (self._last_seq or 0) + 1
        self._gaps += 1
        self._session_vwap_valid = False

        _log.critical(
            "math_engine.seq_gap",
            symbol=self.symbol,
            expected_seq=expected,
            received_seq=received_seq,
            missed=received_seq - expected,
            gaps_this_session=self._gaps,
            action="buffers invalidated; session VWAP void for the rest of the session",
        )
        self._invalidate()

    def _invalidate(self) -> None:
        """Drop all derived state. Allocations are retained and reused."""
        self._prices.clear()
        self._volumes.clear()
        self._candles.clear()
        self._candle_start = None
        self._candle_volume = 0.0
        self._pv_sum = 0.0
        self._v_sum = 0.0
        # The cumulative baseline is meaningless across a gap: the next delta would silently
        # absorb every print we missed.
        self._last_cum_volume = None

    def _fold_candle(self, ts_epoch: float, price: float, volume: float) -> None:
        """Accumulate the tick into the current 5-minute candle, closing it on a boundary."""
        bucket = math.floor(ts_epoch / CANDLE_SECONDS) * CANDLE_SECONDS

        if self._candle_start is None:
            self._open_candle(bucket, price, volume)
            return

        if bucket > self._candle_start:
            self._candles.push(
                start_epoch=self._candle_start,
                open_=self._candle_open,
                high=self._candle_high,
                low=self._candle_low,
                close=self._candle_close,
                volume=self._candle_volume,
            )
            self._open_candle(bucket, price, volume)
            return

        if price > self._candle_high:
            self._candle_high = price
        if price < self._candle_low:
            self._candle_low = price
        self._candle_close = price
        self._candle_volume += volume

    def _open_candle(self, bucket: float, price: float, volume: float) -> None:
        self._candle_start = bucket
        self._candle_open = price
        self._candle_high = price
        self._candle_low = price
        self._candle_close = price
        self._candle_volume = volume

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def tainted(self) -> bool:
        """True while a gap's effects can still reach a rolling indicator.

        Clears once ``recovery_ticks`` clean ticks have arrived — the point at which no
        pre-gap sample remains in the EMA window. :attr:`session_vwap_valid` does not clear.
        """
        return self._gaps > 0 and len(self._prices) < self._recovery_ticks

    @property
    def session_vwap_valid(self) -> bool:
        """False once any gap has occurred. Lost prints cannot be recovered."""
        return self._session_vwap_valid

    @property
    def last_volume_delta(self) -> float:
        """This print's size after differencing (``0.0`` until a second tick arrives).

        Exposed so downstream consumers — the risk gate's z-score accumulator — can reuse
        the exact per-print volume the aggregator computed instead of re-implementing the
        cumulative-to-delta conversion and possibly disagreeing with it.
        """
        return self._last_volume_delta

    @property
    def session_vwap(self) -> float:
        """Exact session VWAP from running accumulators, or ``NaN``.

        Computed incrementally rather than over the ring buffer: the buffer holds the last
        2000 ticks, which after 09:20 is a *rolling* window, not the session. Anchoring the
        day's entries to a rolling VWAP that is labelled session VWAP would be a silent and
        expensive error.
        """
        if not self._session_vwap_valid or self._v_sum <= 0.0:
            return math.nan
        return self._pv_sum / self._v_sum

    @property
    def gaps_detected(self) -> int:
        return self._gaps

    @property
    def tick_count(self) -> int:
        return len(self._prices)

    @property
    def candle_count(self) -> int:
        return len(self._candles)

    @property
    def ltp(self) -> float:
        return self._ltp

    def atr(self) -> float:
        """Wilder ATR over completed 5-minute candles, or ``NaN`` if not yet available."""
        return float(
            calculate_wilder_atr(
                self._candles.highs,
                self._candles.lows,
                self._candles.closes,
                self._atr_period,
            )
        )

    def snapshot(self) -> IndicatorSnapshot:
        """Compute every indicator over the current buffers.

        Views handed to the kernels are contiguous slices of the pre-allocated arrays, so this
        performs no allocation beyond the returned dataclass.
        """
        prices: Any = self._prices.view()
        volumes: Any = self._volumes.view()

        # One fused pass over the price buffer for every EMA period (single JIT dispatch),
        # writing into the pre-allocated scratch — not one kernel call per period.
        calculate_emas(prices, self._ema_periods_arr, self._ema_out)
        emas = tuple(float(value) for value in self._ema_out)

        return IndicatorSnapshot(
            symbol=self.symbol,
            ts_epoch=self._last_ts,
            ltp=self._ltp,
            session_vwap=self.session_vwap,
            rolling_vwap=float(calculate_vwap(prices, volumes)),
            ema_periods=self._ema_periods,
            emas=emas,
            atr_5m=self.atr(),
            obi=self._obi,
            obi_weighted=self._obi_weighted,
            tick_count=len(self._prices),
            candle_count=len(self._candles),
            tainted=self.tainted,
            session_vwap_valid=self._session_vwap_valid,
            gaps_detected=self._gaps,
        )

    def reset_session(self) -> None:
        """Clear everything for a new trading day, including the gap history.

        VWAP is session-anchored and resets at 09:15 IST (CLAUDE.md §3.1).
        """
        self._invalidate()
        self._last_seq = None
        self._gaps = 0
        self._session_vwap_valid = True
        self._ltp = math.nan
        self._last_ts = 0.0
        self._obi = 0.0
        self._obi_weighted = 0.0

    def seed_bars(self, bars: Sequence[HistoricalBar]) -> None:
        """Pre-seed the rolling buffers with historical OHLCV bars.

        Called from the 09:05 IST pre-market boot path
        (:mod:`tachyon.math_engine.warmup`) once
        :meth:`tachyon.ingestion.angel_adapter.AngelOneWebSocketClient.fetch_historical_candles`
        has returned three days of 5-minute candles. The point is to remove the one-hour cold
        start the math engine would otherwise pay for the 14-period Daily ATR, the 20-period
        EMA, and the rolling volume baseline, by populating the same buffers a live session
        would produce — so the very first tick at 09:15 lands on a fully warm indicator set.

        The seeding mirrors the live tick path's effects without inventing a fake sequence:

        * Each bar is pushed into the candle ring so ATR, which reads ``_candles.highs`` /
          ``.lows`` / ``.closes``, has its baseline ready.
        * Each bar's close is appended to the tick ring, giving the EMA fused kernel
          (:func:`calculate_emas`) the lookback it needs.
        * Each bar's volume is appended to the volume ring — the rolling VWAP baseline.
        * The aggregator's last-seen LTP and timestamp are set to the final bar, so the
          snapshot's ``ltp`` and ``ts_epoch`` are meaningful even before the first live tick.

        Session VWAP is **not** touched: the pre-seeded data is from prior days and must not
        contaminate the running session accumulator that opens at 09:15. ``_session_vwap_valid``
        stays True and ``_pv_sum``/``_v_sum`` stay zero, so the first real tick seeds the
        session VWAP exactly as it would without pre-seeding.

        Sequence-based gap detection is also untouched: the pre-seeded bars have no sequence
        number, and the first live tick will seed ``_last_seq`` itself.

        Args:
            bars: chronological (oldest first) historical candles. Empty input is a no-op —
                the aggregator stays cold and the live session warms it from the first tick.
        """
        if not bars:
            return
        for bar in bars:
            self._candles.push(
                start_epoch=float(bar.start_epoch),
                open_=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            )
            self._prices.append(float(bar.close))
            self._volumes.append(float(bar.volume))
        last = bars[-1]
        self._ltp = float(last.close)
        self._last_ts = float(last.start_epoch)

    def __repr__(self) -> str:
        return (
            f"TickAggregator(symbol={self.symbol!r}, ticks={len(self._prices)}, "
            f"candles={len(self._candles)}, gaps={self._gaps})"
        )
