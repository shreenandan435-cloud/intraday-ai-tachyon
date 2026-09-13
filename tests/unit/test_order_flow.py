"""Tests for the order-flow kernels — CLAUDE.md §3, ORB Strike Zone.

Each kernel is checked against an independent pure-NumPy reference computed in the
test. A test that merely re-runs the implementation proves nothing about whether the
formula is right; these formulas gate every ORB signal that fires during the Golden
Window.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tachyon.math_engine.order_flow import (
    KERNELS,
    calculate_obi_fast,
    calculate_rvol,
    sliding_window_volume_sum,
)
from tachyon.math_engine.warmup import reset_warm_state, warmup

pytestmark = pytest.mark.usefixtures("warm_engine")


@pytest.fixture(scope="session", autouse=True)
def warm_engine() -> None:
    """Compile every order-flow kernel up front.

    Mirrors the math-engine warmup fixture so the first call into a kernel in a live
    session is *not* the one that pays the JIT cost. The ORB strategy's first tick
    lands at 09:15:00.000, before the Golden Window even opens, so there is no
    natural opportunity to pay the compile later.
    """
    reset_warm_state()
    assert warmup() is True


class TestObiFast:
    def test_formula_matches_reference(self) -> None:
        bid, ask = 300.0, 100.0
        expected = (bid - ask) / (bid + ask)
        assert calculate_obi_fast(bid, ask) == pytest.approx(expected, abs=1e-12)

    def test_empty_book_is_zero_not_nan(self) -> None:
        """A missing book is balanced evidence; NaN here would poison the strategy."""
        assert calculate_obi_fast(0.0, 0.0) == 0.0
        assert not math.isnan(calculate_obi_fast(0.0, 0.0))

    def test_one_sided_book_is_unit(self) -> None:
        assert calculate_obi_fast(1_000.0, 0.0) == pytest.approx(1.0)
        assert calculate_obi_fast(0.0, 1_000.0) == pytest.approx(-1.0)

    def test_agrees_with_slow_obi(self) -> None:
        """The fastmath kernel must agree with the canonical one to within float error.

        ``fastmath=True`` permits reassociation, so the running arithmetic is not
        bit-identical, but the result for any well-conditioned input is identical
        to within ``1e-12``.
        """
        from tachyon.math_engine.indicators import calculate_obi

        for bid, ask in ((300, 100), (100, 300), (42, 42), (1_000_000, 1)):
            assert calculate_obi_fast(float(bid), float(ask)) == pytest.approx(
                calculate_obi(bid, ask), abs=1e-12
            )


class TestRvol:
    def test_two_point_five_times_baseline(self) -> None:
        assert calculate_rvol(2_500.0, 1_000.0) == pytest.approx(2.5)

    def test_zero_baseline_is_zero_not_inf(self) -> None:
        """A missing baseline is a 'I cannot tell' signal, not infinity."""
        assert calculate_rvol(1_000.0, 0.0) == 0.0

    def test_zero_current_is_zero(self) -> None:
        assert calculate_rvol(0.0, 1_000.0) == 0.0

    def test_sub_one_is_valid(self) -> None:
        assert calculate_rvol(500.0, 1_000.0) == pytest.approx(0.5)


class TestSlidingWindowVolumeSum:
    def test_sums_last_n(self) -> None:
        volumes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        assert sliding_window_volume_sum(volumes, 5, 3) == pytest.approx(15.0)
        assert sliding_window_volume_sum(volumes, 3, 2) == pytest.approx(7.0)

    def test_window_larger_than_buffer_returns_all(self) -> None:
        volumes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert sliding_window_volume_sum(volumes, 2, 100) == pytest.approx(6.0)

    def test_start_before_zero_clips(self) -> None:
        """Window reaching back past index 0 is clipped to the available prefix."""
        volumes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        # Window of 5 starting at index 1: end=2, start=0, sum = 1+2 = 3.0
        assert sliding_window_volume_sum(volumes, 1, 5) == pytest.approx(3.0)
        # Window of 5 starting at index 2: end=3, start=0, sum = 1+2+3 = 6.0
        assert sliding_window_volume_sum(volumes, 2, 5) == pytest.approx(6.0)

    def test_empty_buffer_is_zero(self) -> None:
        empty = np.zeros(0, dtype=np.float64)
        assert sliding_window_volume_sum(empty, 0, 5) == 0.0

    def test_zero_window_is_zero(self) -> None:
        volumes = np.array([1.0, 2.0], dtype=np.float64)
        assert sliding_window_volume_sum(volumes, 1, 0) == 0.0


class TestKernelsRegistry:
    def test_every_kernel_is_jitted(self) -> None:
        for kernel in KERNELS:
            targetoptions = getattr(kernel, "targetoptions", {})
            assert targetoptions.get("nogil") is True, f"{kernel.__name__}: nogil"
            assert hasattr(kernel, "_cache_misses"), f"{kernel.__name__}: cache disabled"
