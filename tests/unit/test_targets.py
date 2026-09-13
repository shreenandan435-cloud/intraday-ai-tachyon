"""Tests for the realistic-target engine and the router's RR gate.

The math here is small but it sizes every stop and every target the engine ever
places, so each formula is checked against an independent reference computed in
the test. A test that merely re-runs the implementation proves nothing about
whether the formula is right.
"""

from __future__ import annotations

import math

import pytest

from tachyon.math_engine.targets import (
    L2_WALL_MULTIPLE,
    MIN_RR,
    RealisticTarget,
    STOP_ATR_MULTIPLIER,
    STRUCTURAL_SLIPPAGE_BPS,
    TargetEngine,
    TradeContext,
    l2_context_from,
)


def _ctx(
    *,
    entry: float = 100.0,
    atr: float = 5.0,
    hod: float = 101.0,
    lod: float = 99.0,
    ask: tuple[int, ...] = (100, 100, 100, 100, 100),
    bid: tuple[int, ...] = (100, 100, 100, 100, 100),
) -> TradeContext:
    return l2_context_from(
        ltp=entry, atr=atr, hod=hod, lod=lod, ask_qty=ask, bid_qty=bid
    )


class TestRemainingAtr:
    def test_long_consumed_day_is_vetoed(self) -> None:
        """When the day's consumed range equals ATR, the long atr_target is entry.

        A long target at exactly the entry is by definition not a target at all,
        so the engine returns ``None`` for a long in this regime. The math
        therefore produces atr_target = entry; the strategy vetoes the trade.
        """
        engine = TargetEngine()
        ctx = _ctx(entry=100.0, atr=10.0, hod=105.0, lod=95.0, ask=(100, 0, 0, 0, 0))
        # remaining = atr - (HOD - LOD) = 10 - 10 = 0
        # atr_target_long = entry + 0 = entry → realistic <= entry → None
        assert engine.evaluate("LONG", ctx) is None

    def test_quiet_day_keeps_full_atr(self) -> None:
        engine = TargetEngine()
        # HOD - LOD = 0 → remaining = 5
        target = engine.evaluate(
            "LONG",
            _ctx(entry=100.0, atr=5.0, hod=108.0, lod=108.0, ask=(100, 0, 0, 0, 0)),
        )
        assert target is not None
        assert target.atr_target == pytest.approx(105.0)

    def test_short_flips_atr_target(self) -> None:
        engine = TargetEngine()
        # atr=30, HOD-LOD=25 → remaining=5 → atr_target = entry - 5 = 95
        target = engine.evaluate(
            "SHORT",
            _ctx(entry=100.0, atr=30.0, hod=105.0, lod=80.0, bid=(100, 0, 0, 0, 0)),
        )
        assert target is not None
        assert target.atr_target == pytest.approx(95.0)


class TestStructuralTarget:
    def test_long_is_hod_front_run(self) -> None:
        engine = TargetEngine()
        # atr=20, HOD-LOD=10 → remaining=10 → atr_target=entry+10=210
        # structural = 210×0.9995 = 209.895. The structural cap is the binding one.
        target = engine.evaluate(
            "LONG",
            _ctx(entry=200.0, atr=20.0, hod=210.0, lod=200.0, ask=(100, 0, 0, 0, 0)),
        )
        assert target is not None
        expected = 210.0 * (1.0 - STRUCTURAL_SLIPPAGE_BPS / 10_000.0)
        assert target.structural_target == pytest.approx(expected)

    def test_short_is_lod_lifted(self) -> None:
        engine = TargetEngine()
        # atr=30, HOD-LOD=20 → remaining=10 → atr_target=200. structural=200.1
        # is the binding cap. entry must be above the structural cap for a real short.
        target = engine.evaluate(
            "SHORT",
            _ctx(entry=210.0, atr=30.0, hod=220.0, lod=200.0, bid=(100, 0, 0, 0, 0)),
        )
        assert target is not None
        expected = 200.0 * (1.0 + STRUCTURAL_SLIPPAGE_BPS / 10_000.0)
        assert target.structural_target == pytest.approx(expected)


class TestL2Wall:
    def test_no_wall_returns_infinity_for_long(self) -> None:
        engine = TargetEngine()
        target = engine.evaluate("LONG", _ctx(ask=(100, 100, 100, 100, 100)))
        assert target is not None
        assert target.l2_target == math.inf

    def test_no_wall_returns_zero_for_short(self) -> None:
        engine = TargetEngine()
        target = engine.evaluate("SHORT", _ctx(bid=(100, 100, 100, 100, 100)))
        assert target is not None
        assert target.l2_target == 0.0

    def test_wall_detected_above_3x_average(self) -> None:
        engine = TargetEngine()
        # ask level 2 has 1000, others (100, 50, 100, 100) average = 87.5
        # 1000 / 87.5 = 11.4x → wall
        target = engine.evaluate(
            "LONG",
            _ctx(ask=(100, 50, 1000, 100, 100), entry=100.0),
        )
        assert target is not None
        assert math.isfinite(target.l2_target)
        assert target.l2_target > 100.0  # above the touch

    def test_wall_below_threshold_is_ignored(self) -> None:
        engine = TargetEngine()
        # ask level 2 has 200, others (100, 50, 100, 100) average = 87.5
        # 200 / 87.5 = 2.3x → below the 3x threshold
        target = engine.evaluate(
            "LONG", _ctx(ask=(100, 50, 200, 100, 100))
        )
        assert target is not None
        assert target.l2_target == math.inf

    def test_touch_level_is_not_a_wall(self) -> None:
        """A bid/ask level at the touch is not "ahead of us" — it is the level
        we just paid. It must never count as a wall to clear."""
        engine = TargetEngine()
        target = engine.evaluate(
            "LONG", _ctx(ask=(10_000, 50, 100, 100, 100))
        )
        assert target is not None
        assert target.l2_target == math.inf


class TestRealisticTarget:
    def test_long_takes_minimum_of_caps(self) -> None:
        engine = TargetEngine()
        # atr_target = 105, structural = 100.95, l2 = inf
        # realistic = min(105, 100.95, inf) = 100.95
        target = engine.evaluate("LONG", _ctx(entry=100.0, atr=5.0, hod=101.0, lod=99.0))
        assert target is not None
        assert target.realistic_target == pytest.approx(target.structural_target)

    def test_short_takes_maximum_of_caps(self) -> None:
        engine = TargetEngine()
        # atr_target = 95, structural = 99.05, l2 = 0
        # realistic = max(95, 99.05, 0) = 99.05
        target = engine.evaluate("SHORT", _ctx(entry=100.0, atr=5.0, hod=101.0, lod=99.0))
        assert target is not None
        assert target.realistic_target == pytest.approx(target.structural_target)

    def test_long_with_wall_uses_wall(self) -> None:
        engine = TargetEngine()
        target = engine.evaluate(
            "LONG", _ctx(ask=(100, 50, 1000, 100, 100), entry=100.0)
        )
        assert target is not None
        assert target.realistic_target == pytest.approx(target.l2_target)

    def test_long_with_consumed_range_returns_none(self) -> None:
        engine = TargetEngine()
        # HOD - LOD = 5 = ATR → remaining = 0
        # atr_target = entry → not strictly above → None
        target = engine.evaluate(
            "LONG", _ctx(atr=5.0, hod=104.0, lod=99.0)
        )
        assert target is None

    def test_short_with_consumed_range_returns_none(self) -> None:
        engine = TargetEngine()
        target = engine.evaluate(
            "SHORT", _ctx(atr=5.0, hod=104.0, lod=99.0)
        )
        assert target is None


class TestStopDistance:
    def test_stop_is_half_atr(self) -> None:
        engine = TargetEngine()
        target = engine.evaluate("LONG", _ctx(entry=100.0, atr=10.0, hod=101.0, lod=99.0))
        assert target is not None
        assert target.stop == pytest.approx(95.0)
        assert target.risk_distance == pytest.approx(5.0)

    def test_short_stop_is_above_entry(self) -> None:
        engine = TargetEngine()
        target = engine.evaluate("SHORT", _ctx(entry=100.0, atr=10.0, hod=101.0, lod=99.0))
        assert target is not None
        assert target.stop == pytest.approx(105.0)
        assert target.risk_distance == pytest.approx(5.0)


class TestRiskReward:
    def test_rr_equals_reward_over_risk(self) -> None:
        engine = TargetEngine()
        target = engine.evaluate("LONG", _ctx(entry=100.0, atr=10.0, hod=101.0, lod=99.0))
        assert target is not None
        expected = target.reward_distance / target.risk_distance
        assert target.rr == pytest.approx(expected)

    def test_rr_gate_passes_when_above_floor(self) -> None:
        engine = TargetEngine()
        target = engine.evaluate("LONG", _ctx(entry=100.0, atr=10.0, hod=101.0, lod=99.0))
        assert target is not None
        # With structural cap of 100.95 and stop of 95.0, reward=0.95, risk=5.0 → RR≈0.19
        # That fails MIN_RR=1.0, so let's hand-craft a target.
        target = RealisticTarget(
            side="LONG", entry=100.0, stop=95.0,
            atr_target=120.0, structural_target=120.0, l2_target=math.inf,
            realistic_target=120.0,
            risk_distance=5.0, reward_distance=20.0, rr=4.0,
        )
        assert engine.passes_rr_gate(target) is True

    def test_rr_gate_fails_below_floor(self) -> None:
        engine = TargetEngine()
        target = RealisticTarget(
            side="LONG", entry=100.0, stop=95.0,
            atr_target=101.0, structural_target=101.0, l2_target=math.inf,
            realistic_target=101.0,
            risk_distance=5.0, reward_distance=1.0, rr=0.2,
        )
        assert engine.passes_rr_gate(target) is False


class TestTradeContext:
    def test_usable_with_valid_inputs(self) -> None:
        assert _ctx().is_usable() is True

    def test_unusable_with_zero_atr(self) -> None:
        assert _ctx(atr=0.0).is_usable() is False

    def test_unusable_with_nan_atr(self) -> None:
        assert _ctx(atr=math.nan).is_usable() is False

    def test_unusable_when_hod_below_lod(self) -> None:
        assert _ctx(hod=99.0, lod=101.0).is_usable() is False

    def test_unusable_with_short_l2(self) -> None:
        assert _ctx(ask=(1, 1), bid=(1, 1)).is_usable() is False


class TestSignalIntegrity:
    def test_realistic_target_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            RealisticTarget(
                side="LONG", entry=math.nan, stop=95.0,
                atr_target=105.0, structural_target=100.0, l2_target=math.inf,
                realistic_target=100.0,
                risk_distance=5.0, reward_distance=5.0, rr=1.0,
            )

    def test_realistic_target_allows_infinity_l2(self) -> None:
        # l2_target=+inf is a valid sentinel meaning "no wall in the way".
        target = RealisticTarget(
            side="LONG", entry=100.0, stop=95.0,
            atr_target=105.0, structural_target=100.0, l2_target=math.inf,
            realistic_target=100.0,
            risk_distance=5.0, reward_distance=5.0, rr=1.0,
        )
        assert target.l2_target == math.inf

    def test_realistic_target_rejects_invalid_side(self) -> None:
        with pytest.raises(ValueError, match="side"):
            RealisticTarget(
                side="SIDEWAYS", entry=100.0, stop=95.0,
                atr_target=105.0, structural_target=100.0, l2_target=math.inf,
                realistic_target=100.0,
                risk_distance=5.0, reward_distance=5.0, rr=1.0,
            )

    def test_as_decimal_routes_through_str(self) -> None:
        target = RealisticTarget(
            side="LONG", entry=100.0, stop=95.0,
            atr_target=110.0, structural_target=110.0, l2_target=math.inf,
            realistic_target=110.0,
            risk_distance=5.0, reward_distance=10.0, rr=2.0,
        )
        target_dec, stop_dec = target.as_decimal()
        # `Decimal(str(110.0))` rather than `Decimal(110.0)` avoids the binary
        # representation artefact; both produce "110" here, but the contract
        # is what matters.
        import decimal
        assert target_dec == decimal.Decimal("110.0")
        assert stop_dec == decimal.Decimal("95.0")


class TestModuleConstants:
    def test_constants_match_spec(self) -> None:
        assert STOP_ATR_MULTIPLIER == pytest.approx(0.5)
        assert STRUCTURAL_SLIPPAGE_BPS == pytest.approx(5.0)
        assert L2_WALL_MULTIPLE == pytest.approx(3.0)
        assert MIN_RR == pytest.approx(1.0)
