"""JIT warmup — CLAUDE.md §3.

Numba compiles on first call, not at import. That compilation takes hundreds of milliseconds
to seconds. Paying it on the first *live* tick would mean the engine's slowest computation of
the entire session happens exactly when the market opens — so the first tick is the one tick
guaranteed to be processed late. CLAUDE.md §3 states it plainly: cold-JIT on a live tick is a
bug.

:func:`warmup` drives every kernel with dummy arrays of the production dtypes, forcing
compilation up front, and verifies the results are sane. It doubles as a self-test: if a
kernel returns garbage on known inputs, the process should not reach the market at all.

``cache=True`` on the kernels persists compiled artefacts to ``__pycache__``, so only the
first run after a code change pays the full cost; later boots reload from disk.

The application must not connect to ZeroMQ until this returns ``True``. That is enforced —
:class:`~tachyon.math_engine.core.TickAggregator` refuses to construct while cold, so no
market data can reach an uncompiled kernel.
"""

from __future__ import annotations

import math
import time
from typing import Final

import numpy as np

from tachyon.core.logger import get_logger
from tachyon.math_engine.indicators import (
    KERNELS,
    calculate_depth_weighted_obi,
    calculate_ema,
    calculate_emas,
    calculate_obi,
    calculate_vwap,
    calculate_wilder_atr,
)
from tachyon.math_engine.order_flow import (
    KERNELS as ORDER_FLOW_KERNELS,
)
from tachyon.math_engine.order_flow import (
    calculate_obi_fast,
    calculate_rvol,
    sliding_window_volume_sum,
)

_log = get_logger(__name__)

_warm: bool = False

#: Enough samples to exercise both the seeding and the recursive branch of every kernel.
_SAMPLES: Final[int] = 64


class EngineNotWarmError(RuntimeError):
    """A kernel path was reached before :func:`warmup` completed successfully."""


def _dummy_series(n: int = _SAMPLES) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic OHLCV with known, finite properties."""
    base = np.linspace(100.0, 110.0, n, dtype=np.float64)
    highs = base + 0.5
    lows = base - 0.5
    volumes = np.full(n, 100.0, dtype=np.float64)
    return base, highs, lows, volumes


def warmup(*, force: bool = False) -> bool:
    """Compile and sanity-check every JIT kernel. Idempotent.

    Args:
        force: recompile even if already warm. For benchmarks and tests.

    Returns:
        ``True`` if every kernel compiled and returned a sane value. ``False`` means the
        process must not trade — the caller should refuse to connect and exit.
    """
    global _warm
    if _warm and not force:
        return True

    started = time.perf_counter()
    closes, highs, lows, volumes = _dummy_series()

    bid_prices = np.array([99.95, 99.90, 99.85, 99.80, 99.75], dtype=np.float64)
    bid_qtys = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
    ask_prices = np.array([100.05, 100.10, 100.15, 100.20, 100.25], dtype=np.float64)
    ask_qtys = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)

    try:
        vwap = calculate_vwap(closes, volumes)
        ema = calculate_ema(closes, 21)
        ema_periods = np.array([9, 21, 50], dtype=np.int64)
        emas_fused = np.empty(ema_periods.shape[0], dtype=np.float64)
        calculate_emas(closes, ema_periods, emas_fused)
        atr_default = calculate_wilder_atr(highs, lows, closes)
        atr_explicit = calculate_wilder_atr(highs, lows, closes, 14)
        obi = calculate_obi(100, 100)
        weighted = calculate_depth_weighted_obi(bid_prices, bid_qtys, ask_prices, ask_qtys)

        # Also compile the insufficient-data branches, so the NaN paths are not themselves
        # a cold-start cost the first time a symbol is short of history.
        empty = np.zeros(0, dtype=np.float64)
        emas_empty = np.empty(ema_periods.shape[0], dtype=np.float64)
        calculate_vwap(empty, empty)
        calculate_ema(closes, 10_000)
        calculate_emas(empty, ema_periods, emas_empty)
        calculate_wilder_atr(highs[:2], lows[:2], closes[:2], 14)
        calculate_obi(0, 0)
        calculate_depth_weighted_obi(empty, empty, empty, empty)

        # Order-flow kernels consumed by the ORB strategy. These run on the live tick
        # path during the 09:20-09:45 window; a cold compile at 09:20:01 is exactly
        # the failure the warmup routine exists to prevent.
        obi_fast = calculate_obi_fast(300.0, 100.0)
        rvol_fast = calculate_rvol(2_500.0, 1_000.0)
        sliding_fast = sliding_window_volume_sum(closes, int(closes.shape[0]) - 1, 5)
    except Exception as exc:  # noqa: BLE001 - a failed warmup must be reported, not raised
        hint = (
            "set NUMBA_CACHE_DIR to a writable directory (the kernel cache locator could "
            "not be resolved)"
            if "cache" in str(exc).lower() or "locator" in str(exc).lower()
            else "inspect the error above"
        )
        _log.critical(
            "math_engine.warmup_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            remedy=hint,
            exc_info=True,
        )
        _warm = False
        return False

    checks = {
        "vwap": math.isclose(vwap, float(np.mean(closes)), rel_tol=1e-9),
        "ema_finite": math.isfinite(ema),
        "ema_in_range": float(closes.min()) <= ema <= float(closes.max()),
        # The fused multi-period kernel must agree BIT-FOR-BIT with the per-period kernel:
        # it is the same recurrence with the dispatch fused, so any divergence means the
        # fusion changed the arithmetic — and every EMA-derived signal would silently shift.
        "emas_fused_bit_identical": all(
            fused == calculate_ema(closes, int(period))
            for fused, period in zip(emas_fused, ema_periods, strict=True)
        ),
        "emas_fused_empty_is_nan": all(math.isnan(value) for value in emas_empty),
        "atr_positive": math.isfinite(atr_default) and atr_default > 0.0,
        "atr_default_matches_explicit": math.isclose(atr_default, atr_explicit, rel_tol=1e-12),
        "obi_balanced_is_zero": obi == 0.0,
        "weighted_obi_balanced_is_zero": math.isclose(weighted, 0.0, abs_tol=1e-12),
        # ORB kernels: same self-test discipline. Each must produce a known value on
        # a known input; a wrong number here would propagate straight into a trade.
        "obi_fast_correct": math.isclose(obi_fast, 0.5, abs_tol=1e-12),
        "rvol_fast_correct": math.isclose(rvol_fast, 2.5, abs_tol=1e-12),
        "sliding_window_sum_correct": math.isclose(
            sliding_fast, float(closes[-5:].sum()), rel_tol=1e-12
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        _log.critical(
            "math_engine.warmup_self_test_failed",
            failed=failed,
            action="refusing to trade — kernels are producing wrong values",
        )
        _warm = False
        return False

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    hits, misses = _cache_stats()
    _warm = True
    _log.info(
        "math_engine.warm",
        elapsed_ms=round(elapsed_ms, 1),
        checks=len(checks),
        cache_hits=hits,
        cache_misses=misses,
        from_cache=misses == 0,
    )
    return True


def _cache_stats() -> tuple[int, int]:
    """Numba on-disk cache hits and misses across every kernel.

    Read from the dispatcher's counters rather than inferred from elapsed time: a wall-clock
    threshold cannot distinguish "recompiled" from "loaded from cache on a busy machine", and
    a silently failing cache would otherwise look like a slow boot rather than a defect.
    Falls back to ``(0, 0)`` if a future numba drops these attributes.
    """
    hits = 0
    misses = 0
    for kernel in (*KERNELS, *ORDER_FLOW_KERNELS):
        hits += sum(getattr(kernel, "_cache_hits", {}).values())
        misses += sum(getattr(kernel, "_cache_misses", {}).values())
    return hits, misses


def is_warm() -> bool:
    """True once :func:`warmup` has succeeded in this process."""
    return _warm


def require_warm() -> None:
    """Raise unless the engine is warm.

    Guards every entry point that could route a live tick into an uncompiled kernel.

    Raises:
        EngineNotWarmError: warmup has not run, or did not succeed.
    """
    if not _warm:
        raise EngineNotWarmError(
            "Math engine is cold. Call tachyon.math_engine.warmup() and check it returns "
            "True before connecting to ZeroMQ (CLAUDE.md §3) — a cold JIT compile on the "
            "first live tick is a bug."
        )


def reset_warm_state() -> None:
    """Forget that warmup ran. Test-support only; does not discard compiled code."""
    global _warm
    _warm = False
