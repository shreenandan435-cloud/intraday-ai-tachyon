"""Numba JIT indicator kernels — CLAUDE.md §3.1.

Every function here is a free function over flat, C-contiguous ``float64`` arrays. No pandas,
no Python objects, no classes, no dicts inside the JIT boundary.

Undefined results return ``NaN``
--------------------------------
When an indicator has insufficient data it returns ``np.nan``, never ``0.0``. This is a safety
decision, not a stylistic one: a VWAP of ``0.0`` makes ``price > vwap`` trivially true and
would manufacture long signals out of an empty buffer, whereas every comparison against
``NaN`` is false, so an undefined indicator produces no signal at all. Callers must test with
``math.isnan`` and treat it as "not ready".

Order Book Imbalance is the documented exception: an empty book returns ``0.0``, meaning
*balanced*, per CLAUDE.md §3.1.

Why ``fastmath`` is off
-----------------------
CLAUDE.md §3 originally specified ``fastmath=True``. It is deliberately **not** used here.
``fastmath`` sets LLVM's ``nnan``/``ninf`` flags, which license the compiler to assume no NaN
ever appears — and this module uses NaN as a load-bearing "indicator undefined" sentinel.
Under those flags, a NaN comparison becomes undefined behaviour, so a not-ready indicator
could silently start producing signals. Measured, the flag buys a few microseconds against a
50 µs budget we already meet by an order of magnitude. Correctness wins.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

_F64 = npt.NDArray[np.float64]


@njit(cache=True, nogil=True)
def calculate_vwap(prices: _F64, volumes: _F64) -> float:
    """Volume-weighted average price.

    ``VWAP = sum(price_i * volume_i) / sum(volume_i)``

    Args:
        prices: trade prices, oldest first.
        volumes: **per-print** volumes, index-aligned with ``prices``.

            Not cumulative session volume. Feeding the exchange's running total here produces
            a heavily back-weighted average that is not a VWAP.
            :class:`~tachyon.math_engine.core.TickAggregator` differences the cumulative field
            before it reaches this function.

    Returns:
        VWAP, or ``NaN`` if there is no traded volume to weight by.
    """
    n = min(prices.shape[0], volumes.shape[0])
    if n == 0:
        return np.nan

    pv_sum = 0.0
    v_sum = 0.0
    for i in range(n):
        pv_sum += prices[i] * volumes[i]
        v_sum += volumes[i]

    if v_sum <= 0.0:
        return np.nan
    return pv_sum / v_sum


@njit(cache=True, nogil=True)
def calculate_ema(prices: _F64, period: int) -> float:
    """Exponential moving average, seeded with the SMA of the first ``period`` values.

    ``alpha = 2 / (period + 1)``, then ``EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}``.

    Seeding from an SMA rather than from ``prices[0]`` removes the startup bias a single
    first print would otherwise inject.

    Returns:
        The final EMA, or ``NaN`` if fewer than ``period`` samples are available.
    """
    n = prices.shape[0]
    if period < 1 or n < period:
        return np.nan

    seed = 0.0
    for i in range(period):
        seed += prices[i]
    ema = seed / period

    alpha = 2.0 / (period + 1.0)
    for i in range(period, n):
        ema = alpha * prices[i] + (1.0 - alpha) * ema
    return ema


@njit(cache=True, nogil=True)
def calculate_wilder_atr(highs: _F64, lows: _F64, closes: _F64, period: int = 14) -> float:
    """Average True Range using **Wilder's** smoothing — the source of truth for all order
    geometry (CLAUDE.md §6.1).

    ``TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)``

    Seeded with the arithmetic mean of the first ``period`` true ranges, then
    ``ATR_t = ((period - 1) * ATR_{t-1} + TR_t) / period``.

    This is Wilder's smoothing, *not* an EMA of TR. An EMA with ``alpha = 2/(period+1)``
    reacts roughly twice as fast and would produce a materially tighter stop than intended —
    every position's risk is sized off this number, so the distinction is not academic.

    Args:
        highs, lows, closes: index-aligned 5-minute candle series, oldest first.
        period: Wilder period, 14 by default.

    Returns:
        ATR, or ``NaN`` if fewer than ``period + 1`` candles are available. The extra candle
        is required because the first true range needs a previous close.
    """
    n = min(highs.shape[0], lows.shape[0], closes.shape[0])
    if period < 1 or n < period + 1:
        return np.nan

    # TR is defined from index 1 onward; seed over the first `period` of them.
    seed = 0.0
    for i in range(1, period + 1):
        previous_close = closes[i - 1]
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - previous_close)
        low_close = abs(lows[i] - previous_close)
        true_range = high_low
        if high_close > true_range:
            true_range = high_close
        if low_close > true_range:
            true_range = low_close
        seed += true_range
    atr = seed / period

    for i in range(period + 1, n):
        previous_close = closes[i - 1]
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - previous_close)
        low_close = abs(lows[i] - previous_close)
        true_range = high_low
        if high_close > true_range:
            true_range = high_close
        if low_close > true_range:
            true_range = low_close
        atr = ((period - 1) * atr + true_range) / period

    return atr


@njit(cache=True, nogil=True)
def calculate_obi(bid_qty: int, ask_qty: int) -> float:
    """Top-of-book Order Book Imbalance.

    ``OBI = (bid_qty - ask_qty) / (bid_qty + ask_qty)``, bounded to ``[-1, +1]``.
    Positive means bid-heavy (buying pressure).

    Returns:
        The imbalance, or ``0.0`` — meaning *balanced* — when the book is empty. This is the
        one kernel that returns zero rather than NaN for an undefined input, per
        CLAUDE.md §3.1: a missing book is genuinely neutral evidence, and NaN here would
        poison a confluence score that is otherwise valid.
    """
    total = bid_qty + ask_qty
    if total <= 0:
        return 0.0
    return (bid_qty - ask_qty) / total


@njit(cache=True, nogil=True)
def calculate_depth_weighted_obi(
    bid_prices: _F64, bid_qtys: _F64, ask_prices: _F64, ask_qtys: _F64
) -> float:
    """Depth-weighted OBI with distance decay across the L2 ladder.

    Size resting far from the touch is far less likely to be executed than size at the touch,
    so weighting every level equally overstates the influence of deep orders — which are also
    the easiest to spoof. Each level is therefore discounted by its distance from the mid:

    ``distance_i = |price_i - mid| / half_spread``  (dimensionless, in half-spreads)
    ``weight_i   = 1 / (1 + distance_i)``

    Measuring distance in half-spreads rather than rupees keeps the kernel scale-free: it
    behaves identically on a ₹100 stock and a ₹3000 one, and needs no tuned constant.

    If the book is crossed, one-sided, or empty, the price scale is meaningless and the kernel
    falls back to positional decay ``weight_i = 1 / (1 + i)`` (CLAUDE.md §3.1).

    Args:
        bid_prices, bid_qtys, ask_prices, ask_qtys: L2 ladders, index 0 at the touch.

    Returns:
        Weighted imbalance in ``[-1, +1]``, or ``0.0`` when there is no size at all.
    """
    n = min(bid_prices.shape[0], bid_qtys.shape[0], ask_prices.shape[0], ask_qtys.shape[0])
    if n == 0:
        return 0.0

    best_bid = bid_prices[0]
    best_ask = ask_prices[0]
    half_spread = (best_ask - best_bid) * 0.5
    mid = (best_ask + best_bid) * 0.5

    # A crossed, empty or locked book gives no usable price scale.
    use_price_distance = best_bid > 0.0 and best_ask > 0.0 and half_spread > 0.0

    weighted_bid = 0.0
    weighted_ask = 0.0
    for i in range(n):
        if use_price_distance:
            bid_weight = 1.0 / (1.0 + abs(bid_prices[i] - mid) / half_spread)
            ask_weight = 1.0 / (1.0 + abs(ask_prices[i] - mid) / half_spread)
        else:
            bid_weight = 1.0 / (1.0 + i)
            ask_weight = bid_weight
        weighted_bid += bid_weight * bid_qtys[i]
        weighted_ask += ask_weight * ask_qtys[i]

    total = weighted_bid + weighted_ask
    if total <= 0.0:
        return 0.0
    return (weighted_bid - weighted_ask) / total


#: Every JIT kernel in this module, for the warmup routine to walk.
KERNELS = (
    calculate_vwap,
    calculate_ema,
    calculate_wilder_atr,
    calculate_obi,
    calculate_depth_weighted_obi,
)
