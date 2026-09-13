"""VWAP z-score filter tests — Track 1 anti-chase guardrail (CLAUDE.md §4, check 12).

The accumulators are checked against an independent pure-NumPy weighted-moment reference
computed in the test, never against values copied from the implementation. A test that
merely re-runs the implementation proves nothing about whether the formula is right, and
this filter vetoes real entries.

The risk-gate section proves the wiring contract: a LONG whose latest print sits more than
+2σ above session VWAP is refused under ``VWAP_OVEREXTENDED``, a SHORT in the identical
state is not, and no path through the filter can raise into the caller.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from tachyon.core.clock import IST, ManualClock
from tachyon.core.state import DailyLock, StateMachine, TradingState
from tachyon.ipc.monitor import FeedMonitor
from tachyon.math.vwap_filter import (
    DEFAULT_Z_THRESHOLD,
    VWAPZScoreFilter,
    prewarm_vwap_filter,
)
from tachyon.math_engine import warmup
from tachyon.risk.engine import RiskEngine, VetoReason
from tachyon.risk.tracker import PnLTracker, PositionRegistry

THRESHOLD = DEFAULT_Z_THRESHOLD


@pytest.fixture(scope="session", autouse=True)
def _warm_engine() -> None:
    """Compile every JIT kernel once for the whole session."""
    assert warmup() is True


def _reference_moments(prices: np.ndarray, volumes: np.ndarray) -> tuple[float, float]:
    """Independent volume-weighted mean and population sigma via NumPy moments."""
    vwap = float(np.average(prices, weights=volumes))
    second_moment = float(np.average(prices * prices, weights=volumes))
    return vwap, math.sqrt(max(0.0, second_moment - vwap * vwap))


# ──────────────────────────────────────────────────────────────────────────────
# O(1) accumulators against analytical values
# ──────────────────────────────────────────────────────────────────────────────


class TestAccumulators:
    def test_known_analytical_series(self) -> None:
        """prices=[100, 102], volumes=[1, 3]: VWAP=101.5, σ=√0.75, z₂=(102−101.5)/σ."""
        filt = VWAPZScoreFilter(THRESHOLD)

        first = filt.process_tick(100.0, 1.0)
        # One print: zero dispersion → inside the bands by contract, VWAP is the price.
        assert first[0] is False
        assert first[1] == pytest.approx(100.0)
        assert first[2] == pytest.approx(0.0)
        assert first[3] == pytest.approx(0.0)

        _, vwap, sigma, z = filt.process_tick(102.0, 3.0)
        assert vwap == pytest.approx((100.0 * 1.0 + 102.0 * 3.0) / 4.0)
        expected_sigma = math.sqrt(max(0.0, (100.0**2 * 1.0 + 102.0**2 * 3.0) / 4.0 - 101.5**2))
        assert sigma == pytest.approx(expected_sigma)
        assert z == pytest.approx((102.0 - 101.5) / expected_sigma)

    def test_matches_numpy_weighted_moments_at_every_step(self) -> None:
        """Every step compared against a from-scratch NumPy recomputation of the prefix."""
        rng = np.random.default_rng(42)
        prices = rng.normal(1500.0, 4.0, 300)
        volumes = rng.integers(1, 900, 300).astype(np.float64)

        filt = VWAPZScoreFilter(THRESHOLD)
        for i in range(len(prices)):
            got_over, got_vwap, got_sigma, got_z = filt.process_tick(
                float(prices[i]), float(volumes[i])
            )

            vwap, sigma = _reference_moments(prices[: i + 1], volumes[: i + 1])
            assert got_vwap == pytest.approx(vwap, rel=1e-12, abs=1e-9)
            assert got_sigma == pytest.approx(sigma, rel=1e-12, abs=1e-9)

            if sigma <= 0.0:
                assert got_z == pytest.approx(0.0)
                assert got_over is False
            else:
                expected_z = (float(prices[i]) - vwap) / sigma
                assert got_z == pytest.approx(expected_z, rel=1e-12, abs=1e-9)
                assert got_over == (expected_z > THRESHOLD)

    def test_accumulators_are_running_sums_not_a_window(self) -> None:
        """O(1) accumulation must remember every print: a late spike cannot be diluted
        away by subsequent quiet ticks the way a fixed rolling window would."""
        filt = VWAPZScoreFilter(THRESHOLD)
        for _ in range(500):
            filt.process_tick(100.0, 10.0)
        filt.process_tick(200.0, 10.0)  # one enormous print
        for _ in range(500):
            filt.process_tick(100.0, 10.0)

        _, vwap, _, _ = filt.process_tick(100.0, 10.0)
        # 1001 prints at 100 and one at 200: VWAP = (1001·100 + 200)/1002 ≈ 100.0998.
        # A 2000-tick rolling window would instead report ≈ 100.05 — close, but a
        # cumulative-volume check separates them decisively.
        assert vwap == pytest.approx((1001 * 100.0 + 200.0) / 1002.0, rel=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Degenerate inputs — nothing may divide by zero or poison the session
# ──────────────────────────────────────────────────────────────────────────────


class TestDegenerateInputs:
    def test_zero_volume_on_empty_session_is_safe(self) -> None:
        filt = VWAPZScoreFilter(THRESHOLD)
        for _ in range(10):
            over, vwap, sigma, z = filt.process_tick(100.0, 0.0)
            assert over is False
            assert math.isnan(vwap)
            assert math.isnan(sigma)
            assert math.isnan(z)
        assert filt.cum_vol == 0.0

    def test_zero_volume_mid_session_advances_nothing(self) -> None:
        """A zero-volume print is scored against the session but changes nothing.

        Analytical: [100×10, 102×30] → VWAP=101.5, σ=√0.75; probing at 105 must return
        z=(105−101.5)/√0.75 and leave every accumulator bit-identical.
        """
        filt = VWAPZScoreFilter(THRESHOLD)
        filt.process_tick(100.0, 10.0)
        filt.process_tick(102.0, 30.0)
        before = (filt.cum_vol, filt.cum_pv, filt.cum_pv2)

        over, vwap, sigma, z = filt.process_tick(105.0, 0.0)

        assert over is True  # +4σ: scored, but...
        assert (filt.cum_vol, filt.cum_pv, filt.cum_pv2) == before  # ...never accumulated
        assert vwap == pytest.approx(101.5)
        assert sigma == pytest.approx(math.sqrt(0.75))
        assert z == pytest.approx((105.0 - 101.5) / math.sqrt(0.75))

    def test_negative_volume_is_ignored_like_zero(self) -> None:
        """A feed reset producing a negative delta must not shrink the denominators."""
        filt = VWAPZScoreFilter(THRESHOLD)
        filt.process_tick(100.0, 10.0)
        filt.process_tick(101.0, 20.0)
        good = (filt.cum_vol, filt.cum_pv, filt.cum_pv2)
        filt.process_tick(999.0, -5.0)
        assert (filt.cum_vol, filt.cum_pv, filt.cum_pv2) == good

    def test_single_print_has_zero_dispersion_not_nan(self) -> None:
        """The very first real print makes the z ratio 0/0; it must read as inside-band."""
        filt = VWAPZScoreFilter(THRESHOLD)
        over, vwap, sigma, z = filt.process_tick(2750.0, 500.0)
        assert over is False
        assert vwap == pytest.approx(2750.0)
        assert sigma == 0.0
        assert z == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Band membership — the anti-chase semantics
# ──────────────────────────────────────────────────────────────────────────────


class TestOverextensionBands:
    @staticmethod
    def _baseline(seed: int, n: int = 400) -> VWAPZScoreFilter:
        rng = np.random.default_rng(seed)
        filt = VWAPZScoreFilter(THRESHOLD)
        for price, volume in zip(rng.normal(100.0, 0.1, n), np.full(n, 10.0), strict=True):
            filt.process_tick(float(price), float(volume))
        return filt

    def test_within_band_returns_false(self) -> None:
        filt = self._baseline(7)
        _, vwap, sigma, _ = filt.process_tick(100.0, 10.0)
        near_mean = vwap + 0.5 * sigma
        over, _, _, z = filt.process_tick(near_mean, 10.0)
        assert over is False
        assert z < THRESHOLD

    def test_beyond_band_returns_true(self) -> None:
        filt = self._baseline(7)
        _, vwap, sigma, _ = filt.process_tick(100.0, 10.0)
        spike = vwap + 5.0 * sigma
        over, _, _, z = filt.process_tick(spike, 10.0)
        assert over is True
        assert z > THRESHOLD

    def test_comparison_is_strictly_greater_than(self) -> None:
        """A print scoring exactly at the threshold passes; only beyond it vetoes.

        Deterministic closed form: [100×1, 102×3, 103×1] → VWAP=101.8, σ=√0.96,
        z₃=1.2/√0.96. Bit-identical arithmetic lets us sit exactly on the boundary.
        """
        cum_vol, cum_pv, cum_pv2 = 5.0, 509.0, 51_821.0
        vwap = cum_pv / cum_vol
        sigma = math.sqrt(max(0.0, cum_pv2 / cum_vol - vwap * vwap))
        z_exact = (103.0 - vwap) / sigma

        at_boundary = VWAPZScoreFilter(z_exact)
        at_boundary.process_tick(100.0, 1.0)
        at_boundary.process_tick(102.0, 3.0)
        over, _, _, z = at_boundary.process_tick(103.0, 1.0)
        assert z == z_exact  # same IEEE operations, same order → bit-identical
        assert over is False

        just_inside = VWAPZScoreFilter(math.nextafter(z_exact, -math.inf))
        just_inside.process_tick(100.0, 1.0)
        just_inside.process_tick(102.0, 3.0)
        over, _, _, _ = just_inside.process_tick(103.0, 1.0)
        assert over is True

    def test_custom_threshold_is_respected(self) -> None:
        """Identical data, two thresholds: the tighter band flags, the looser one passes."""
        prefix_prices = np.append(np.full(200, 100.0), 100.05)
        volumes = np.full(201, 10.0)

        loose = VWAPZScoreFilter(THRESHOLD)
        tight = VWAPZScoreFilter(0.5)
        for filt in (loose, tight):
            for price, volume in zip(prefix_prices, volumes, strict=True):
                filt.process_tick(float(price), float(volume))

        vwap, sigma = _reference_moments(prefix_prices, volumes)
        target = vwap + 0.8 * sigma

        loose_over, _, _, loose_z = loose.process_tick(target, 10.0)
        tight_over, _, _, tight_z = tight.process_tick(target, 10.0)

        assert loose_z == pytest.approx(tight_z)  # same data, same score
        assert loose_over is False and loose_z <= THRESHOLD
        assert tight_over is True and tight_z > 0.5


class TestPureScore:
    def test_score_does_not_mutate_the_accumulators(self) -> None:
        filt = VWAPZScoreFilter(THRESHOLD)
        filt.process_tick(100.0, 10.0)
        filt.process_tick(102.0, 30.0)
        before = (filt.cum_vol, filt.cum_pv, filt.cum_pv2)

        for price in (99.0, 101.5, 110.0, 90.0):
            filt.score(price)

        assert (filt.cum_vol, filt.cum_pv, filt.cum_pv2) == before

    def test_score_matches_process_tick_statistics(self) -> None:
        """Both views must report identical session statistics; only inclusion differs."""
        rng = np.random.default_rng(5)
        filt = VWAPZScoreFilter(THRESHOLD)
        prices = rng.normal(500.0, 1.0, 100)
        for price in prices:
            filt.process_tick(float(price), 20.0)

        scored = filt.score(float(prices[-1]))
        processed = filt.process_tick(float(prices[-1]), 0.0)  # zero-volume: no mutation

        assert scored[1] == pytest.approx(processed[1])
        assert scored[2] == pytest.approx(processed[2])
        assert scored[3] == pytest.approx(processed[3])
        assert scored[0] == processed[0]

    def test_gate_verdict_tracks_the_price_being_judged_not_the_last_print(self) -> None:
        """score() judges its argument: a below-VWAP last print must not mask a spike."""
        filt = VWAPZScoreFilter(THRESHOLD)
        rng = np.random.default_rng(9)
        for price in rng.normal(100.0, 0.05, 300):
            filt.process_tick(float(price), 50.0)

        filt.process_tick(99.9, 10.0)  # last print sits *below* VWAP
        over_spike, _, _, z_spike = filt.score(101.0)  # ...but the judged price spikes

        assert over_spike is True
        assert z_spike > THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# Session reset
# ──────────────────────────────────────────────────────────────────────────────


class TestSessionReset:
    def test_reset_clears_all_state(self) -> None:
        filt = VWAPZScoreFilter(THRESHOLD)
        for i in range(50):
            filt.process_tick(100.0 + i * 0.01, 25.0)
        assert filt.cum_vol > 0.0 and filt.cum_pv > 0.0 and filt.cum_pv2 > 0.0

        filt.reset()

        assert filt.cum_vol == 0.0
        assert filt.cum_pv == 0.0
        assert filt.cum_pv2 == 0.0
        assert filt.z_threshold == THRESHOLD, "the configured threshold survives the day"

    def test_reset_behaviourally_equals_a_fresh_filter(self) -> None:
        rng = np.random.default_rng(11)
        prefix = [
            (float(p), float(v))
            for p, v in zip(rng.normal(99.0, 0.3, 80), np.full(80, 7), strict=True)
        ]
        reused = VWAPZScoreFilter(THRESHOLD)
        for price, volume in prefix:
            reused.process_tick(price, volume)
        reused.reset()

        fresh = VWAPZScoreFilter(THRESHOLD)
        for price, volume in prefix:
            assert reused.process_tick(price, volume) == fresh.process_tick(price, volume)

    def test_prewarm_is_idempotent(self) -> None:
        assert prewarm_vwap_filter() is True
        assert prewarm_vwap_filter() is True


# ──────────────────────────────────────────────────────────────────────────────
# Risk-gate integration — the reason the filter exists
# ──────────────────────────────────────────────────────────────────────────────


class _Harness:
    """A fully wired risk stack whose every input can be perturbed."""

    def __init__(self, tmp_path) -> None:
        self.clock = ManualClock(wall=datetime(2026, 8, 10, 11, 0, tzinfo=IST), mono=1000.0)
        machine = StateMachine(TradingState.ACTIVE, clock=self.clock)
        lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=self.clock)
        pnl = PnLTracker(machine, daily_lock=lock, clock=self.clock)
        monitor = FeedMonitor(clock=self.clock)
        monitor.record()  # feed is alive by default
        self.engine = RiskEngine(machine, pnl, monitor, PositionRegistry(), clock=self.clock)


class TestRiskGateIntegration:
    @staticmethod
    def _feed_quiet_tape(engine: RiskEngine, n: int = 400) -> None:
        """A tight tape around ₹100: tiny realised σ, so a modest pop is many sigmas."""
        rng = np.random.default_rng(3)
        for price in rng.normal(100.0, 0.02, n):
            engine.observe_tick("RELIANCE", float(price), 100.0)

    def test_long_beyond_two_sigma_is_vetoed_without_raising(self, tmp_path) -> None:
        harness = _Harness(tmp_path)
        self._feed_quiet_tape(harness.engine)

        # +0.30 % above VWAP: well inside the 1.5 % percentage rule...
        ltp, vwap = 100.30, 100.0
        assert (ltp - vwap) / vwap * 100.0 < 1.5
        # ...but dozens of sigmas above it. Only the z-score clause can catch this.
        decision = harness.engine.evaluate("RELIANCE", direction="LONG", quote=(ltp, vwap))

        assert not decision.allowed
        assert decision.reason is VetoReason.VWAP_OVEREXTENDED
        assert "σ" in decision.detail, "detail must identify the z-score clause fired"

    def test_short_at_the_same_extreme_is_not_blocked(self, tmp_path) -> None:
        """The rule guards buying the top; shorting into exhaustion is another decision."""
        harness = _Harness(tmp_path)
        self._feed_quiet_tape(harness.engine)
        decision = harness.engine.evaluate("RELIANCE", direction="SHORT", quote=(100.30, 100.0))
        assert decision.allowed

    def test_no_accumulator_passes_the_clause(self, tmp_path) -> None:
        """No ticks observed → no evidence → the clause cannot veto what it has not seen."""
        harness = _Harness(tmp_path)
        decision = harness.engine.evaluate("RELIANCE", direction="LONG", quote=(100.30, 100.0))
        assert decision.allowed

    def test_within_band_long_is_allowed(self, tmp_path) -> None:
        harness = _Harness(tmp_path)
        self._feed_quiet_tape(harness.engine)
        decision = harness.engine.evaluate("RELIANCE", direction="LONG", quote=(100.001, 100.0))
        assert decision.allowed

    def test_non_finite_ticks_are_discarded_and_never_raise(self, tmp_path) -> None:
        harness = _Harness(tmp_path)
        self._feed_quiet_tape(harness.engine)
        harness.engine.observe_tick("RELIANCE", float("nan"), 100.0)
        harness.engine.observe_tick("RELIANCE", float("inf"), 100.0)
        harness.engine.observe_tick("RELIANCE", 100.0, float("nan"))

        decision = harness.engine.evaluate("RELIANCE", direction="LONG", quote=(100.001, 100.0))
        assert decision.allowed, "garbage input must not veto and must not poison the sums"

    def test_session_reset_reopens_the_gate(self, tmp_path) -> None:
        harness = _Harness(tmp_path)
        self._feed_quiet_tape(harness.engine)
        assert not harness.engine.evaluate(
            "RELIANCE", direction="LONG", quote=(100.30, 100.0)
        ).allowed

        harness.engine.reset_vwap_session()

        assert harness.engine.evaluate(
            "RELIANCE", direction="LONG", quote=(100.30, 100.0)
        ).allowed, "a new session starts with no evidence of extension"

    def test_observe_tick_survives_a_cold_prewarm(self, tmp_path) -> None:
        """If compilation failed at boot the feed becomes a silent no-op, never a fault."""
        harness = _Harness(tmp_path)
        harness.engine._vwap_z_ready = False
        harness.engine.observe_tick("RELIANCE", 100.0, 10.0)  # must not raise
        assert harness.engine.evaluate("RELIANCE", direction="LONG", quote=(100.30, 100.0))
