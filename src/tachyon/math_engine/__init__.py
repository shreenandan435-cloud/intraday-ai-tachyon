"""Numba JIT math engine — CLAUDE.md §3.

Every kernel is an ``@njit(cache=True, nogil=True)`` free function over flat NumPy arrays.
No pandas, no Python objects inside JIT. Zero allocation in the tick path: ring buffers are
pre-allocated and mirrored so a contiguous view is always a plain slice.

``warmup()`` must return ``True`` before the process connects to ZeroMQ — cold-JIT on a live
tick is a bug. :class:`~tachyon.math_engine.core.TickAggregator` refuses to construct while
cold, so market data cannot reach an uncompiled kernel.

Modules:
  buffers.py     RingBuffer (mirrored, zero-copy contiguous views) and CandleBuffer
  indicators.py  VWAP, EMA, Wilder ATR, top-of-book OBI, depth-weighted OBI
  core.py        TickAggregator — seq-gap detection, cumulative-volume differencing,
                 5-minute candle folding, indicator snapshots
  warmup.py      compile-and-self-test at boot

Undefined indicators return ``NaN``, never ``0.0`` — a zeroed VWAP would manufacture signals.
OBI is the documented exception and returns ``0.0`` (balanced) for an empty book.
"""

from __future__ import annotations

from tachyon.math_engine.preseed import (
    AggregatorHost,
    bars_from_rows,
    preseed_aggregator,
    preseed_universe,
)
from tachyon.math_engine.warmup import (
    EngineNotWarmError,
    is_warm,
    require_warm,
    reset_warm_state,
    warmup,
)

# NOTE: re-exporting the ``warmup`` *function* shadows the ``warmup`` *module* as an attribute
# of this package. ``from tachyon.math_engine import warmup`` gives you the function (the
# intended API), but ``import tachyon.math_engine.warmup as m`` then binds the function too,
# so ``m.require_warm`` raises AttributeError. Import submodule members directly —
# ``from tachyon.math_engine.warmup import require_warm`` — and this never bites.
__all__ = [
    "AggregatorHost",
    "EngineNotWarmError",
    "bars_from_rows",
    "is_warm",
    "preseed_aggregator",
    "preseed_universe",
    "require_warm",
    "reset_warm_state",
    "warmup",
]
