"""Benchmark the Numba math engine against the CLAUDE.md §3 budget.

Target: **< 50 µs** for a full indicator refresh of one symbol, at production buffer sizes
(2000 ticks, 500 five-minute candles).

Run::

    .venv\\Scripts\\python.exe scripts/bench_math_engine.py

Exits non-zero if the budget is missed, so it can gate a release.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tachyon.core.logger import configure_logging  # noqa: E402
from tachyon.ipc.schemas import OrderBook, Tick  # noqa: E402
from tachyon.math_engine import warmup  # noqa: E402
from tachyon.math_engine.core import TickAggregator  # noqa: E402
from tachyon.math_engine.indicators import (  # noqa: E402
    calculate_depth_weighted_obi,
    calculate_ema,
    calculate_obi,
    calculate_vwap,
    calculate_wilder_atr,
)

BUDGET_US = 50.0
ITERATIONS = 2_000
TICK_CAPACITY = 2_000
CANDLE_CAPACITY = 500


def _percentiles(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    return (
        statistics.median(ordered),
        ordered[int(len(ordered) * 0.99)],
        ordered[-1],
    )


def _time(label: str, fn, *args, iterations: int = ITERATIONS) -> float:  # type: ignore[no-untyped-def]
    fn(*args)  # exclude any first-call cost from the sample
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - start) * 1e6)
    p50, p99, worst = _percentiles(samples)
    print(f"  {label:<34s} p50 {p50:8.2f} us   p99 {p99:8.2f} us   max {worst:8.2f} us")
    return p50


def main() -> int:
    configure_logging(role="bench", json_lines=False, pretty_console=True)

    print("Warming JIT kernels...")
    started = time.perf_counter()
    if not warmup():
        print("FAIL: warmup did not succeed — kernels are producing wrong values.")
        return 1
    print(f"  warmup completed in {(time.perf_counter() - started) * 1000:.1f} ms\n")

    rng = np.random.default_rng(20260810)
    prices = 2400 + np.cumsum(rng.normal(0, 0.5, TICK_CAPACITY))
    volumes = rng.integers(1, 500, TICK_CAPACITY).astype(np.float64)
    closes = 2400 + np.cumsum(rng.normal(0, 2.0, CANDLE_CAPACITY))
    highs = closes + rng.uniform(0.5, 5.0, CANDLE_CAPACITY)
    lows = closes - rng.uniform(0.5, 5.0, CANDLE_CAPACITY)

    bid_prices = np.array([2399.95, 2399.90, 2399.85, 2399.80, 2399.75])
    ask_prices = np.array([2400.05, 2400.10, 2400.15, 2400.20, 2400.25])
    bid_qtys = np.array([100.0, 250.0, 300.0, 420.0, 500.0])
    ask_qtys = np.array([120.0, 200.0, 340.0, 400.0, 560.0])

    print(f"Individual kernels ({TICK_CAPACITY} ticks / {CANDLE_CAPACITY} candles):")
    _time("calculate_vwap", calculate_vwap, prices, volumes)
    _time("calculate_ema(9)", calculate_ema, prices, 9)
    _time("calculate_ema(21)", calculate_ema, prices, 21)
    _time("calculate_ema(50)", calculate_ema, prices, 50)
    _time("calculate_wilder_atr(14)", calculate_wilder_atr, highs, lows, closes, 14)
    _time("calculate_obi", calculate_obi, 100, 200)
    _time(
        "calculate_depth_weighted_obi",
        calculate_depth_weighted_obi,
        bid_prices,
        bid_qtys,
        ask_prices,
        ask_qtys,
    )

    # Full refresh through the aggregator, at capacity, exactly as the strategy calls it.
    aggregator = TickAggregator(
        "RELIANCE", tick_capacity=TICK_CAPACITY, candle_capacity=CANDLE_CAPACITY
    )
    base = 1_786_000_000.0
    cumulative = 0
    sequence = 0
    # 60 s between ticks fills the 500-candle buffer, so ATR is timed at full depth rather
    # than against a handful of candles.
    for i in range(TICK_CAPACITY * 2):
        cumulative += int(rng.integers(1, 400))
        sequence += 1
        aggregator.on_tick(
            Tick(
                token="2885",
                ltp=float(prices[i % TICK_CAPACITY]),
                volume=cumulative,
                ts_epoch=base + i * 60.0,
                seq=sequence,
            )
        )
    aggregator.on_orderbook(
        OrderBook(
            token="2885",
            bid_price=(2399.95, 2399.90, 2399.85, 2399.80, 2399.75),
            bid_qty=(100, 250, 300, 420, 500),
            ask_price=(2400.05, 2400.10, 2400.15, 2400.20, 2400.25),
            ask_qty=(120, 200, 340, 400, 560),
            ts_epoch=base,
        )
    )

    print(
        f"\nFull indicator block "
        f"(ticks={aggregator.tick_count}, candles={aggregator.candle_count}):"
    )
    snapshot_p50 = _time("TickAggregator.snapshot()", aggregator.snapshot)

    # Pre-built contiguous ticks: reusing one seq would trip gap detection on every call and
    # measure the invalidation path instead of the ingest path.
    ingest_iterations = 2_000
    feed = [
        Tick(
            token="2885",
            ltp=2400.0 + (i % 20) * 0.05,
            volume=cumulative + i + 1,
            ts_epoch=base + (TICK_CAPACITY * 2 + i) * 60.0,
            seq=sequence + i + 1,
        )
        for i in range(ingest_iterations + 1)
    ]
    aggregator.on_tick(feed[0])  # consume the first so the timed run stays contiguous
    ingest_samples = []
    for tick in feed[1:]:
        start = time.perf_counter()
        aggregator.on_tick(tick)
        ingest_samples.append((time.perf_counter() - start) * 1e6)
    p50, p99, worst = _percentiles(ingest_samples)
    print(
        f"  {'TickAggregator.on_tick()':<34s} p50 {p50:8.2f} us   "
        f"p99 {p99:8.2f} us   max {worst:8.2f} us"
    )
    if aggregator.gaps_detected:
        print(f"  WARNING: benchmark induced {aggregator.gaps_detected} sequence gaps")

    print()
    if snapshot_p50 <= BUDGET_US:
        print(f"PASS: full refresh {snapshot_p50:.2f} us <= {BUDGET_US:.0f} us budget")
        return 0
    print(f"FAIL: full refresh {snapshot_p50:.2f} us exceeds the {BUDGET_US:.0f} us budget")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
